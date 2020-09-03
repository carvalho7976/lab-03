from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import os

drive_path = '/mnt/sda4/lab-03/meses/'
save_path = '/mnt/sda4/data-aumentations-lab03/'

## Arquivo de entrada
entrada = drive_path + 'test.txt'

arq = open(entrada,'r')
conteudo_entrada = arq.readlines()
arq.close()

## Diretorio da base de dados (imagens)
dir_dataset = drive_path + 'data/'

## Diretorio de destino
dir_destino = save_path + 'libsvm/'

### Cria o diretorio de destino (caso nao exista)
# if not os.path.exists(dir_destino):
# 	os.makedirs(dir_destino)

# Cria arquivo com as caracteristicas
arq_svm = dir_destino + 'out.svm'
file_svm = open(arq_svm, 'w')

# Input resize (minimo=75x75)
img_rows, img_cols = 100, 100

print('Done')

# InceptionV3
# - weights='imagenet' (inicializa pesos pre-treinado na ImageNet)
# - include_top=False (nao inclui as fully-connected layers)
# - input_shape=(299, 299, 3) (DEFAULT) (minimo=75x75)
model = InceptionV3(weights='imagenet', include_top=False)

# Mostra a arquitetura da rede
model.summary()
print ("Loading...")

for i in conteudo_entrada:
  
  nome, classe = i.split()

  img_path = dir_dataset + nome
  print (img_path) ##
  
  img = image.load_img(img_path, target_size=(img_rows,img_cols))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)

  # Passa a imagem pela rede
  inception_features = model.predict(img_data)

  # Flatten
  features_np = np.array(inception_features)
  features_np = features_np.flatten()

  # Salva no formato do libsvm
  file_svm.write(classe+' ')
  for j in range (features_np.size):
    file_svm.write(str(j+1)+':'+str(features_np[j])+' ')
  file_svm.write('\n')

print (features_np.size)
file_svm.close()