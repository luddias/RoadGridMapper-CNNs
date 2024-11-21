import os
import shutil
import pandas as pd

# Função para contar o número total de imagens e máscaras nos diretórios fornecidos
def count_files(directory):
    images = masks = 0
    for filename in os.listdir(directory):
        if filename.startswith('i'):
            images += 1
        elif filename.startswith('r'):
            masks += 1
    return images, masks

# Função para mover imagens e máscaras para os diretórios de destino e gerar o arquivo CSV correspondente
def process_dataset(dataset_dir, output_dir, subset, target_percentage):
    # Criar diretórios necessários
    os.makedirs(os.path.join(output_dir, subset, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, subset, "masks"), exist_ok=True)

    
    # Calculando o número de pastas a serem vasculhadas
    num_folders = len(os.listdir(dataset_dir))
    target_images = int(num_folders*target_percentage)
    print(target_images)

    # DataFrame para armazenar os caminhos das imagens e máscaras movidas
    df = pd.DataFrame(columns=["Image Path", "Mask Path"])
    
    # Movendo imagens e máscaras para o novo dataset
    for subdir, dirs, _ in os.walk(dataset_dir):
        for dir in dirs[:target_images]:
            files = os.listdir(os.path.join(subdir,dir))
            for file in files:
                if file.startswith('i'):
                    # print('DENTRO') 
                    
                    mask = 'r'+file[1:]
                    print(file, mask)
                    shutil.move(os.path.join(subdir, dir, file), os.path.join(output_dir, subset, "images", file))                

                    shutil.move(os.path.join(subdir, dir, mask), os.path.join(output_dir, subset, "masks", mask))

                    image_mask_pairs = [os.path.join(output_dir, subset, "images", file), os.path.join(output_dir, subset, "masks", mask)]
                    
                    df.loc[len(df)] = image_mask_pairs

    # Salvando DataFrame como CSV
    df.to_csv(os.path.join(output_dir, f"{subset}.csv"), index=False)

def rename_test_to_validation(csv_file):
    # Carregar o arquivo CSV
    df = pd.read_csv(csv_file)

    # Alterar todos os endereços da pasta test para validation
    df['Image Path'] = df['Image Path'].str.replace('/test/', '/validation/')
    df['Mask Path'] = df['Mask Path'].str.replace('/test/', '/validation/')

    # Salvar o DataFrame modificado de volta ao CSV
    df.to_csv(csv_file, index=False)


def main():
    # Diretório original do dataset
    dataset_dir = "Highway_dataset_road_mapper_20180204/guarapari"
    # Diretório onde o novo dataset será criado
    output_dir = "DATASET_RGM"

    # Processar o conjunto de treinamento
    # process_dataset(dataset_dir, output_dir, "train", 0.8)

    # Processar o conjunto de teste
    process_dataset(dataset_dir, output_dir, "test", 1)
    # rename_test_to_validation('DATASET_RGM/validation.csv')
    
main()