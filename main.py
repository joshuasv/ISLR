import os
import argparse

import yaml

import globals
from data_prep import preprocess
from train import train

def load_yaml(fname):
    fpath = os.path.join(globals.CONFIG_DPATH, f"{fname}.yaml")
    with open(fpath, 'r') as fid:
        return yaml.load(fid, yaml.SafeLoader)

def load_yaml_preprocessing(fname):
    yaml = load_yaml(fname)
    toret = yaml['preprocessing']
    toret['DEBUG'] = yaml['DEBUG']
    toret['config_fname'] = fname

    return toret

def load_yaml_train(fname):
    toret = {}
    yaml = load_yaml(fname)
    toret['model'] = yaml['model']
    toret['train'] = yaml['train']
    toret['data'] = yaml['data']
    toret['DEBUG'] = yaml['DEBUG']
    toret['SEED'] = yaml['SEED']
    toret['config_fname'] = fname

    return toret

def main(args):
    try:
        f = args.func
        del args.func
        f(**vars(args))
    except:
        raise Exception

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Google ISLR Kaggle competition')
    subparsers = parser.add_subparsers(title='actions')

    parser_data = subparsers.add_parser(
        'data',
        parents=[parser],
        add_help=False,
        description='Preprocess the raw data.',
        help='preprocess the raw data')
    parser_data.add_argument(
        '--raw_data_dpath',
        type=str, 
        help='raw data directory path'
    )
    parser_data.add_argument(
        '--out_root_dpath',
        type=str, 
        help='root preprocessed data output directory path'
    )
    parser_data.add_argument(
        '--config',
        type=load_yaml_preprocessing,
        required=True,
        help='dataset preprocessing configuration YAML file name'
    )
    parser_data.set_defaults(
        raw_data_dpath=globals.DATA_RAW_DPATH,
        out_root_dpath=globals.DATA_PREPROCESSED_DPATH,
        func=preprocess)
                    
    parser_train = subparsers.add_parser(
        'train',
        parents=[parser],
        add_help=False,
        description='Train a deep learning model!',
        help='train a deep learning model')
    parser_train.add_argument(
        '--config',
        type=load_yaml_train,
        required=True,
        help='training configuration YAML file name'
    )
    parser_train.set_defaults(func=train)

    args = parser.parse_args()
    main(args)