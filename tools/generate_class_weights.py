from dataset import Dataset
import argparse
from utils import get_class_weights


# Cria um objeto parser
parser = argparse.ArgumentParser(description='Gerar pesos das classes com base na frequência')

# Adiciona os argumentos que o seu programa aceita
parser.add_argument('--train_csv', type= str, help='Caminho para o csv de treino')
parser.add_argument('--img_shape',  type= int, help='tamanho imagens')
parser.add_argument('-nc', '--num_classes',  type= int, help='Num de classes')


# Faz o parse dos argumentos da linha de comando
args = parser.parse_args()

train_dataset = Dataset(args.train_csv, args.num_classes, args.img_shape)
print('[INFO]Datasets Sucessfully defined!')

# Get the class weights
print('[INFO]Starting to define the class weights...')
get_class_weights(train_dataset, args.img_shape)
