import os
import json
import pprint
import datetime
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from bnaf import *

import functools

from GQ_MS import GQ_MS


def load_dataset(args):

    #convert datasets
    # data = pd.read_csv(r'C:\Users\just\PycharmProjects\BNAF\data\gas\ethylene_methane.txt', delim_whitespace=True, header='infer')
    # data.to_pickle('data/gas/ethylene_methane.pickle')

    tf.random.set_seed(args.manualSeed)
    np.random.seed(args.manualSeed)

    if args.dataset == 'gq_ms_wheat':
        dataset = GQ_MS('GQ_MS/wheat_perms.xlsx')
    elif args.dataset == 'gq_ms_soy':
        dataset = GQ_MS('GQ_MS/soy_perms.xlsx')
    elif args.dataset == 'gq_ms_corn':
        dataset = GQ_MS('GQ_MS/corn_perms.xlsx')
    elif args.dataset == 'gq_ms_canola':
        dataset = GQ_MS('GQ_MS/canola_perms.xlsx')
    elif args.dataset == 'gq_ms_barley':
        dataset = GQ_MS('GQ_MS/barley_perms.xlsx')
    elif args.dataset == 'gq_ms_all':
        dataset = GQ_MS('GQ_MS/all_perms.xlsx')
    else:
        raise RuntimeError()


    dataset_train = dataset.trn.x.astype(np.float32)
    dataset_valid = dataset.val.x.astype(np.float32)
    dataset_test = dataset.tst.x.astype(np.float32)

    args.n_dims = dataset.n_dims

    return dataset_train, dataset_valid, dataset_test


def create_model(args, verbose=False):

    # random.seed(manualSeed)
    # torch.manual_seed(manualSeed)

    tf.random.set_seed(args.manualSeed)
    np.random.seed(args.manualSeed)

    dtype_in = tf.float32

    flows = []
    for f in range(args.flows):
        #build internal layers for a single flow
        layers = []
        for _ in range(args.layers - 1):
            layers.append(MaskedWeight(args.n_dims * args.hidden_dim,
                                       args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in))
            layers.append(Tanh(dtype_in=dtype_in))

        flows.append(
            BNAF(layers = [MaskedWeight(args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in), Tanh(dtype_in=dtype_in)] + \
               layers + \
               [MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims, dtype_in=dtype_in)], \
             res=args.residual if f < args.flows - 1 else None
             )
        )

        if f < args.flows - 1:
            flows.append(Permutation(args.n_dims, 'flip'))

        model = Sequential(flows)
        # params = np.sum(np.sum(p.numpy() != 0) if len(p.numpy().shape) > 1 else p.numpy().shape
        #              for p in model.trainable_variables)[0]
    
    # if verbose:
    #     print('{}'.format(model))
    #     print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(params,
    #         NAF_PARAMS[args.dataset][0] / params, NAF_PARAMS[args.dataset][1] / params, args.n_dims))

    # if args.save and not args.load:
    #     with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
    #         print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(params,
    #             NAF_PARAMS[args.dataset][0] / params, NAF_PARAMS[args.dataset][1] / params, args.n_dims), file=f)
    
    return model

def load_model(args, root, load_start_epoch=False):
    # def f():
    print('Loading model..')
    root.restore(tf.train.latest_checkpoint(args.load or args.path))
    # root.restore(os.path.join(args.load or args.path, 'checkpoint'))
    # if load_start_epoch:
    #     args.start_epoch = tf.train.get_global_step().numpy()
    # return f

# @tf.function
def compute_log_p_x(model, x_mb):
    ## use tf.gradient + tf.convert_to_tensor + tf.GradientTape(persistent=True) to clean up garbage implementation in bnaf.py
    y_mb, log_diag_j_mb = model(x_mb)
    log_p_y_mb = tf.reduce_sum(tfp.distributions.Normal(tf.zeros_like(y_mb), tf.ones_like(y_mb)).log_prob(y_mb), axis=-1)#.sum(-1)
    return log_p_y_mb + log_diag_j_mb

class parser_:
    pass

def main():
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    # tf.compat.v1.enable_eager_execution(config=config)

    # tf.config.experimental_run_functions_eagerly(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    args = parser_()
    args.device = '/cpu:0'  # '/gpu:0'
    args.dataset = 'gq_ms_wheat' #['gas', 'bsds300', 'hepmass', 'miniboone', 'power']
    args.learning_rate = np.float32(1e-2)
    args.batch_dim = 200
    args.clip_norm = 0.1
    args.epochs = 5000
    args.patience = 10
    args.cooldown = 10
    args.decay = 0.5
    args.min_lr = 5e-4
    args.flows = 1
    args.layers = 1
    args.hidden_dim = 3
    args.residual = 'gated'
    args.expname = ''
    args.load = r'C:\Users\just\PycharmProjects\BNAF\checkpoint\gq_ms_wheat_layers1_h3_flows1_gated_2019-07-28-22-39-13'
    #C:\Users\just\PycharmProjects\BNAF\checkpoint\gq_ms_wheat_layers1_h3_flows1_gated_2019-07-28-15-46-02
    args.save = True
    args.tensorboard = 'tensorboard'
    args.manualSeed = 1


    args.path = os.path.join('checkpoint', '{}{}_layers{}_h{}_flows{}{}_{}'.format(
        args.expname + ('_' if args.expname != '' else ''),
        args.dataset, args.layers, args.hidden_dim, args.flows, '_' + args.residual if args.residual else '',
        str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

    print('Loading dataset..')

    data_loader_train, data_loader_valid, data_loader_test = load_dataset(args)

    alldata_1, all_data2 = load_dataset(args)

    if args.save and not args.load:
        print('Creating directory experiment..')
        os.mkdir(args.path)
        with open(os.path.join(args.path, 'args.json'), 'w') as f:
            json.dump(str(args.__dict__), f, indent=4, sort_keys=True)
    
    print('Creating BNAF model..')
    with tf.device(args.device):
        model = create_model(args, verbose=True)

    ### debug
    # data_loader_train_ = tf.contrib.eager.Iterator(data_loader_train)
    # x = data_loader_train_.get_next()
    # a = model(x)

    ## tensorboard and saving
    writer = tf.summary.create_file_writer(os.path.join(args.tensorboard, args.load or args.path))
    writer.set_as_default()
    tf.compat.v1.train.get_or_create_global_step()

    root = None
    print('Creating optimizer..')
    with tf.device(args.device):
        optimizer = tf.optimizers.Adam()
    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.compat.v1.train.get_global_step())

    if args.load:
        load_model(args, root, load_start_epoch=True)


if __name__ == '__main__':
    main()

##"C:\Program Files\Git\bin\sh.exe" --login -i

#### tensorboard --logdir=C:\Users\just\PycharmProjects\BNAF\tensorboard\checkpoint
## http://localhost:6006/

