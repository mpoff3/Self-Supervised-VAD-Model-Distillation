import os
import argparse
import torch
import time
import pickle
import numpy as np

from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import VideoAnomalyDataset_C3D
from models import model

from tqdm import tqdm
from aggregate import remake_video_output, evaluate_auc, remake_video_3d_output

torch.backends.cudnn.benchmark = False

# Config
DATA_DIR = os.environ["VAD_DATASET_PATH"]


def get_configs():
    parser = argparse.ArgumentParser(description="VAD-Jigsaw config")
    parser.add_argument("--val_step", type=int, default=500)
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu_id", type=str, default=0)
    parser.add_argument("--log_date", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--static_threshold", type=float, default=0.3)
    parser.add_argument("--sample_num", type=int, default=5, help="Time length of a window")
    parser.add_argument("--filter_ratio", type=float, default=0.8)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="shanghaitech", choices=['shanghaitech', 'ped2', 'avenue'])
    parser.add_argument("--debug_data", action="store_true")
    parser.add_argument("--prefetch", type=int, default=None)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--sample_step", type=int, default=1, help="Step size for sampling frames during testing")
    parser.add_argument("--model_stride", type=int, default=1, help="sampling rate of the model (do not impact the sample num)")
    parser.add_argument("--out_checkpoints", type=str, default='./checkpoint', help="out folder for the checkpoints")
    parser.add_argument("--model_config", type=str, default='B', help=f"Configuration of the network among the predone config: {list(model.MODEL_CONFIGS.keys())}")
    parser.add_argument('--interpolation_method', type=str, default='max_pool',
                        choices=['max_pool', 'linear', 'nearest', 'gaussian', 'moving_avg', 'none'],
                        help='Method to use for interpolating scores when sample_step > 1')
    parser.add_argument('--filter_method', type=str, default='mean',
                        choices=['mean', 'gaussian', 'median', 'bilateral', 'none'],
                        help='Method to use for 3D filtering of scores')

    args = parser.parse_args()

    args.device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    if args.dataset in ['shanghaitech', 'avenue']:
        args.filter_ratio = 0.8
    elif args.dataset == 'ped2':
        args.filter_ratio = 0.5
    return args


def train(args):
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    print("The running_data : {}".format(running_date))
    for k, v in vars(args).items():
        print("-------------{} : {}".format(k, v))

    # Load Data
    data_dir = f"{DATA_DIR}{args.dataset}_frames/training"
    detect_pkl = f'detect/{args.dataset}_train_detect_result_yolov3.pkl'

    net = model.WideBranchNet(time_length=args.sample_num, num_classes=[args.sample_num ** 2, 81], variant=args.model_config)
    if args.model_stride > 1:
        net = model.WideBranchNet_Strided(time_length=args.sample_num, stride=args.model_stride, nb_patches=9, variant=args.model_config)

    testing_dataset = VideoAnomalyDataset_C3D(f"{DATA_DIR}{args.dataset}_frames/testing",
                                              dataset=args.dataset,
                                              detect_dir=f'detect/{args.dataset}_test_detect_result_yolov3.pkl',
                                              fliter_ratio=args.filter_ratio,
                                              sample_step=args.sample_step,
                                              frame_num=args.sample_num)

    testing_data_loader = DataLoader(testing_dataset, batch_size=args.batch_size, shuffle=False, num_workers=(args.workers+1)//2, drop_last=False,
                                     prefetch_factor=args.prefetch if args.workers > 0 else None, persistent_workers=True if args.workers > 0 else False, pin_memory=True)

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=args.device)
        print('load ' + args.checkpoint)
        try:
            net.load_state_dict(state, strict=True)
        except Exception:
            net.load_state_dict(model.flatten_state_dict(state, net.state_dict()), strict=True)
        net.to(args.device)
        smoothed_auc, smoothed_auc_avg, _ = val(args, testing_data_loader, net)
        exit(0)

    vad_dataset = VideoAnomalyDataset_C3D(data_dir,
                                          dataset=args.dataset,
                                          detect_dir=detect_pkl,
                                          fliter_ratio=args.filter_ratio,
                                          frame_num=args.sample_num,
                                          static_threshold=args.static_threshold,
                                          cache=args.cache,
                                          model_stride=args.model_stride)

    vad_dataloader = DataLoader(vad_dataset, batch_size=args.batch_size, shuffle=True, persistent_workers=True if args.workers > 0 else False,
                                num_workers=args.workers, pin_memory=True, prefetch_factor=args.prefetch if args.workers > 0 else None,
                                )

    net.to(args.device)
    net = net.train()

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(params=net.parameters(), lr=args.lr)

    # Train
    log_dir = './log/{}/'.format(running_date)
    writer = SummaryWriter(log_dir)

    t0 = time.time()
    global_step = 0

    max_acc = -1
    timestamp_in_max = None

    for epoch in tqdm(range(args.epochs)):
        for it, data in enumerate(vad_dataloader):
            video, obj, temp_labels, spat_labels, t_flag = data['video'], data['obj'], data['label'], data["trans_label"], data["temporal"]
            n_temp = t_flag.sum().item()
            if args.debug_data:
                continue
            obj = obj.cuda(args.device, non_blocking=True)

            if args.model_stride > 1:
                temp_labels, _ = net.transform_label(temp_labels)

            temp_labels = temp_labels[t_flag].long().view(-1).cuda(args.device)
            spat_labels = spat_labels[~t_flag].long().view(-1).cuda(args.device)

            temp_logits, spat_logits = net(obj)
            temp_logits = temp_logits[t_flag].view(-1, net.time_length)
            spat_logits = spat_logits[~t_flag].view(-1, 9)
            temp_loss = criterion(temp_logits, temp_labels)
            spat_loss = criterion(spat_logits, spat_labels)
            loss = temp_loss + spat_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train/Loss', loss.item(), global_step=global_step)
            writer.add_scalar('Train/Temporal', temp_loss.item(), global_step=global_step)
            writer.add_scalar('Train/Spatial', spat_loss.item(), global_step=global_step)

            if (global_step + 1) % args.print_interval == 0:
                print("[{}:{}/{}]\tloss: {:.6f} t_loss: {:.6f} s_loss: {:.6f} \ttime: {:.6f}".
                      format(epoch, it + 1, len(vad_dataloader), loss.item(), temp_loss.item(), spat_loss.item(),  time.time() - t0), flush=True)
                t0 = time.time()

            global_step += 1

            if global_step % args.val_step == 0 and epoch >= 5:
                smoothed_auc, smoothed_auc_avg, temp_timestamp = val(args, testing_data_loader, net)
                writer.add_scalar('Test/smoothed_auc', smoothed_auc, global_step=global_step)
                writer.add_scalar('Test/smoothed_auc_avg', smoothed_auc_avg, global_step=global_step)

                if smoothed_auc > max_acc:
                    max_acc = smoothed_auc
                    timestamp_in_max = temp_timestamp
                    save = '{}_{}.pth'.format('best', running_date)
                    os.makedirs(args.out_checkpoints, exist_ok=True)
                    # torch.save(net.state_dict(), os.path.join(args.out_checkpoints, save))
                    with open(os.path.join(args.out_checkpoints, save), 'wb') as f:
                        torch.save(net.state_dict(), f)
                        f.flush()
                        os.fsync(f.fileno())

                print('cur max: ' + str(max_acc) + ' in ' + timestamp_in_max)
                net = net.train()


def val(args, testing_data_loader, net=None):
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    print("The running_date : {}".format(running_date))

    net.eval()

    video_output = {}
    for data in tqdm(testing_data_loader):
        videos = data["video"]
        frames = data["frame"].tolist()
        obj = data["obj"].to(args.device)

        with torch.no_grad():
            temp_logits, spat_logits = net(obj)
            temp_logits = temp_logits.view(-1, net.time_length, net.time_length)
            spat_logits = spat_logits.view(-1, 9, 9)

        spat_probs = F.softmax(spat_logits, -1)
        diag = torch.diagonal(spat_probs, offset=0, dim1=-2, dim2=-1)
        scores = diag.min(-1)[0].cpu().numpy()

        temp_probs = F.softmax(temp_logits, -1)
        diag2 = torch.diagonal(temp_probs, offset=0, dim1=-2, dim2=-1)
        scores2 = diag2.min(-1)[0].cpu().numpy()

        for video_, frame_, s_score_, t_score_ in zip(videos, frames, scores, scores2):
            if video_ not in video_output:
                video_output[video_] = {}
            if frame_ not in video_output[video_]:
                video_output[video_][frame_] = []
            video_output[video_][frame_].append([s_score_, t_score_])

    micro_auc, macro_auc = save_and_evaluate(video_output, running_date, dataset=args.dataset, sample_step=args.sample_step, interpolation_method=args.interpolation_method, filter_method=args.filter_method, name=args.model_config)
    return micro_auc, macro_auc, running_date


def save_and_evaluate(video_output, running_date, dataset='shanghaitech', sample_step=1, interpolation_method='max_pool', filter_method='mean', name=None):
    pickle_path = './log/video_output_ori_{}.pkl'.format(running_date if name is None else name)
    with open(pickle_path, 'wb') as write:
        pickle.dump(video_output, write, pickle.HIGHEST_PROTOCOL)
    if dataset == 'shanghaitech':
        video_output_spatial, video_output_temporal, video_output_complete = remake_video_output(video_output, dataset=dataset)
    else:
        _, _, video_output_complete = remake_video_3d_output(video_output, dataset=dataset, sample_step=sample_step,
                                                             interpolation_method=interpolation_method,
                                                             filter_method=filter_method)
    smoothed_res, smoothed_auc_list = evaluate_auc(video_output_complete, dataset=dataset)
    return smoothed_res.auc, np.mean(smoothed_auc_list)


if __name__ == '__main__':
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    args = get_configs()
    train(args)
    # python main.py --dataset avenue --val_step 100 --print_interval 20 --batch_size 192 --sample_num 7 --epochs 100 --static_threshold 0.2 --sample_step 1 --model_config Mv6
    # python main.py --dataset avenue --val_step 100 --print_interval 20 --batch_size 192 --sample_num 7 --epochs 3 --static_threshold 0.2 --debug_data
    # python main.py --dataset avenue --sample_num 7 --checkpoint ../avenue_92.18.pth --sample_step 1
    # python main.py --dataset avenue --sample_num 7 --checkpoint ../final/Bstride2.pth --sample_step 1  --workers 4
