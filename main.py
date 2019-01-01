import argparse
import os
import tensorflow as tf
from som import som
parser = argparse.ArgumentParser(description='')


parser.add_argument('--dataset', dest='dataset', default='box', help='chosen dataset')
parser.add_argument('--n1', dest='n1', type=int, default=10, help='number of units in each row')
parser.add_argument('--n2', dest='n2', type=int, default=10, help='number of units in each col')
parser.add_argument('--run_id', dest='run_id', type=int, default=0, help='An ID identifying this run')

parser.add_argument('--sigma_i', dest='sigma_i', type=float, default=3, help='Initial value of sigma')
parser.add_argument('--sigma_f', dest='sigma_f', type=float, default=0.1, help='Final Value of sigma')
parser.add_argument('--eps_i', dest='eps_i', type=float, default=0.5, help='Initial learning rate')
parser.add_argument('--eps_f', dest='eps_f', type=float, default=0.005, help='Final learning rate')

parser.add_argument('--ntype', dest='ntype', default="GAUSSIAN", help='nborhood type (GAUSSIAN,CONSTANT)')
parser.add_argument('--plotmode', dest='plotmode', default="CLASS_COLOR", help='plot mode (CLASS_COLOR,CLASS_NOCOLOR)')
parser.add_argument('--gridmode', dest='gridmode', default="RECTANGLE", help='grid type (RECTANGLE,HEXAGON)')
parser.add_argument('--initmode', dest='initmode', default="PCAINIT", help='initialization (RANDOM,PCAINIT)')

parser.add_argument('--n_iter', dest='n_iter', type=int, default=10000, help='total number of iterations')
parser.add_argument('--plot_iter', dest='plot_iter', type=int, default=10000, help='number of iterations between plots')



parser.add_argument('--lr', dest='lr', type=float, default=2e-05, help='learning rate')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=400, help='# images in batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=100000, help='# of epoch')
parser.add_argument('--arch', dest='arch', default="h1024r;h1024r;h1024r;", help='architecture string')

parser.add_argument('--momentum', dest='momentum', action='store_true', help='if True, use momentum for the optimizer')
parser.set_defaults(momentum=True)

parser.add_argument('--lamb', dest='lamb', type=float, default=0, help='parameter for weight regularization')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.9, help='parameter for dropout')
parser.add_argument('--cv', dest='cv', type=int, default=5, help='chosen test fold (1 to 5 isolet, 1 to 10 wine,iris)')



parser.add_argument('--dataset_dir', dest='dataset_dir', default='horse2zebra', help='path of the dataset')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=300, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')

args = parser.parse_args()  


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    with tf.device("/gpu:0"):
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        #tfconfig.gpu_options.allow_growth = True
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8
        with tf.Session(config=tfconfig) as sess:
            model = som(sess, args)
            model.build_model()
            model.train()
            ######model.train(args) if args.phase == 'train' \
            ######    else model.test(args)
        
if __name__ == '__main__':
    tf.app.run()
