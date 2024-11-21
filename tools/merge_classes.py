import numpy as np

def merge_classes(mask, class_values):
    # Inicializar nova máscara com zeros
    new_mask = np.zeros(mask.shape, dtype=np.uint8)

    # Converter class_values para numpy array para vetorização
    class_values = np.array(class_values, dtype=object)

    # Criar uma máscara booleana para cada classe de valores
    for idx, value in enumerate(class_values):
        bool_mask = np.isin(mask, value)
        new_mask[bool_mask] = idx

        # Adicionalmente, tratar o último caso se necessário
        if idx == len(class_values) - 1:
            bool_mask = mask > value[-1]
            new_mask[bool_mask] = idx

    return new_mask
