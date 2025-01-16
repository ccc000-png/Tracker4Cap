import os
import datetime
import argparse
__all__ = ['TotalConfigs', 'get_opts']

def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

def parse_opt():
    parser = argparse.ArgumentParser()
    """
    =========================Sys Settings===========================
    """
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--init_method', default="tcp://localhost:2223")
    parser.add_argument('--distributed', default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")

    """
    =========================General Settings===========================
    """
    parser.add_argument('--loglevel', type=str, default='DEBUG',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--bsz', type=int, default=128, help='batch size')
    parser.add_argument('--sample_numb', type=int, default=15,
                        help='how many frames would you like to sample from a given video')
    parser.add_argument('--clip_name', type=str, default='clip_l14', help='which model you would like to train/test?')
    parser.add_argument('--Track', action='store_true')
    parser.add_argument('--Base', type=int, default=0)
    parser.add_argument('--Age', action='store_true')
    parser.add_argument('--fusion_object', action='store_true')
    parser.add_argument('--fusion_action', action='store_true')

    """
    =========================Data Settings===========================
    """
    parser.add_argument('--dataset', type=str, default='msvd')
    parser.add_argument('--train_type', type=str, default='preprocess')
    parser.add_argument('--data_root', type=str,
                        default='./data', help='all msvd and msrvtt')
    parser.add_argument('--checkpoints_dir', type=str, default='./result/checkpoints')
    parser.add_argument('--save_checkpoints', type=str, default='./result/checkpoints')
    parser.add_argument('--log_dir', type=str, default='./log/experiment')
    parser.add_argument('--log_freq', type=int, default=1)

    """
    =========================Encoder Settings===========================
    """
    parser.add_argument('--visual_dim', type=int, default=768, help='dimention for inceptionresnetv2=2048,clip=512/768')
    parser.add_argument('--object_dim', type=int, default=None, help='if,use pretrained, object dimention for vg_objects')
    parser.add_argument('--track_objects', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)

    """
    =========================Decoder Settings===========================
    """
    parser.add_argument('--max_caption_len', type=int, default=22)
    parser.add_argument('--decoder_type', type=str, default="bert")
    parser.add_argument('--decoder_layers', type=int, default=1)
    """
    =========================Training Settings===========================
    """
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)  # 4
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--schedule', type=str, default="warmup_constant")
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--clip_lr', type=float, default=2e-6)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.95)
    args = parser.parse_args()

    return args

class TotalConfigs:
    def __init__(self, args):
        self.data = DataConfigs(args)
        self.encoder = EncoderConfigs(args)
        self.decoder = DecoderConfigs(args)
        self.train = TrainingConfigs(args)
        self.bert = BertConfigs(args)

        self.num_gpu = args.num_gpu
        self.num_shards = args.num_shards
        self.shard_id = args.shard_id
        self.init_method = args.init_method
        self.distributed = args.distributed
        self.num_workers = args.num_workers
        self.local_rank =args.local_rank

        self.loglevel = args.loglevel
        self.seed = args.seed
        self.bsz = args.bsz
        self.drop_prob = args.drop_prob
        self.sample_numb = args.sample_numb
        self.Track = args.Track
        self.Base = args.Base

        self.Age = args.Age
        if self.Base ==0:
            self.Track = True
            args.Track = True
        if self.Track:
            self.fusion_object = args.fusion_object
        else:
            args.fusion_object = False
            self.fusion_object = args.fusion_object
        if self.Age:
            self.fusion_action = args.fusion_action
        else:
            args.fusion_action = False
            self.fusion_action = args.fusion_action

class DataConfigs:
    def __init__(self, args):
        self.train_type = args.train_type
        self.dataset = args.dataset
        self.data_root = args.data_root
        self.checkpoints_dir = args.checkpoints_dir
        self.log_dir = args.log_dir
        self.logger_file = 'logger.log'
        self.log_freq = args.log_freq

        if args.clip_name=="clip_b16":
            self.clip_weights = "model_zoo/clip_model/ViT-B-16.pt"
        elif args.clip_name=="clip_l14":
            self.clip_weights = "model_zoo/clip_model/ViT-L-14.pt"

        # data root
        self.data_root = os.path.join(self.data_root, self.dataset)
        self.visual_dir = os.path.join(self.data_root, 'visual')
        self.visual_features = os.path.join(self.visual_dir,
                                            '{clip_name}/frame_feature'.format(clip_name=args.clip_name))
        self.object_features = os.path.join(self.visual_dir, '{}_vg_objects_{}.hdf5'.format(args.dataset, '{}'))

        # lang root
        self.language_dir = os.path.join(self.data_root, 'language')

        self.ann_root = os.path.join(self.language_dir, '{dataset}_caption.json'.format(dataset=args.dataset))
        self.video_root = os.path.join('./data/{dataset}'.format(dataset=args.dataset), 'videos')

        # dataset split part
        self.videos_split = os.path.join(self.data_root, 'splits/{}_{}_list.pkl'.format(self.dataset, '{}'))

class BertConfigs:
    def __init__(self, args):
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.lr = args.lr
        self.warmup = args.warmup
        self.schedule = args.schedule
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        self.clip_lr = args.clip_lr
        self.lr_decay_gamma = args.lr_decay_gamma
        self.vocab_size = 49408


class EncoderConfigs:
    def __init__(self, args):
        if args.clip_name=='clip_b16' or args.clip_name=='clip_b32':
            self.visual_dim = 512
        else:
            self.visual_dim = 768
        # self.visual_dim = args.visual_dim
        self.object_dim = args.object_dim
        self.hidden_dim = args.hidden_dim
        self.track_objects = args.track_objects

class DecoderConfigs:
    def __init__(self, args):
        self.max_caption_len = args.max_caption_len
        self.decoder_type =args.decoder_type
        self.decoder_layers = args.decoder_layers

class TrainingConfigs:
    def __init__(self, args):
        self.grad_clip = args.grad_clip
        self.lr = args.lr
        self.max_epochs = args.max_epochs
        self.save_checkpoints = args.save_checkpoints
        self.checkpoints_dir = os.path.join(args.checkpoints_dir,
                                     "{}/Track_{}_AGE_{}_Fusion_Track_{}_Fusion_AGE_{}_{}".format(args.dataset, args.Track, args.Age, args.fusion_object, args.fusion_action, get_timestamp()))
        self.save_checkpoints_path = os.path.join(self.checkpoints_dir,
                                                  '{clip_name}_epochs_{max_epochs}_lr_{lr}_max_objects_{mo}.ckpt'.format(
                                                      clip_name=args.clip_name,
                                                      max_epochs=args.max_epochs,
                                                      lr=args.lr,
                                                      mo=args.track_objects))
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.evaluate_dir = os.path.join(self.checkpoints_dir, 'evaluate.txt')
        self.captions_dir = os.path.join(self.checkpoints_dir, 'gen_captions.txt')
        self.save_freq = args.save_freq

def get_opts():
    args = parse_opt()
    configs = TotalConfigs(args=args)
    return configs