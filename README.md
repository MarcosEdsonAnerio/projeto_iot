# Sistema de Reconhecimento Facial - Projeto IoT
Este repositório é dedicado ao desenvolvimento de um sistema de Reconhecimento Facial Multimodal como parte das atividades acadêmicas na disciplina de Programação Web 2 no Instituto Federal de Rondônia - Campus Ji-Paraná (IFRO). Este projeto visa criar uma solução prática e automatizada para identificação de indivíduos em tempo real utilizando visão computacional e inteligência artificial.

## 📌 Objetivo do Projeto
O objetivo deste projeto é implementar um sistema de reconhecimento facial eficiente que identifica alunos com base em imagens de referência. Utilizando modelos avançados de visão computacional, o sistema é capaz de analisar rostos, reconhecer e nomear indivíduos na tela, e organizar automaticamente as capturas. Ele é projetado para rodar em tempo real, com capacidade de análise em ambientes acadêmicos, como escolas e faculdades, para identificação segura e automática de alunos.

## 🛠️ Tecnologias Utilizadas

Este projeto foi construído com uma combinação de ferramentas de IA e bibliotecas de visão computacional:

**Python** - Linguagem principal para desenvolvimento do sistema

**OpenCV** - Processamento de imagem em tempo real

**InsightFace** - Biblioteca para detecção e reconhecimento facial

**Torch** - Framework para carregamento e execução de modelos pré-treinados de rede neural

**NumPy** - Manipulação e cálculo com arrays, para suporte ao processamento de imagens

**Scikit-Learn** - Cálculo da similaridade entre embeddings faciais usando distância de cosseno

## 📐 Estrutura do Projeto
**images** - Diretório que armazena as imagens de referência de cada indivíduo, organizadas em subpastas com o nome dos alunos.
**unknown** - Diretório para armazenar imagens capturadas de indivíduos não reconhecidos.
**scripts** - Arquivos de código Python responsáveis pela detecção, reconhecimento e armazenamento de imagens.

## ⚙️ Funcionalidades Principais

Detecção Facial em Tempo Real: O sistema captura imagens em tempo real utilizando uma câmera conectada, detectando faces presentes na cena.
Reconhecimento com Base em Embeddings: Ao iniciar, o sistema carrega embeddings faciais de cada aluno usando as imagens presentes na pasta images. Para cada rosto detectado, ele compara com os embeddings carregados para identificar se a pessoa é um aluno conhecido.
Identificação e Exibição do Nome: Caso o rosto seja reconhecido, o nome do aluno é exibido em tempo real acima da cabeça, com uma indicação de similaridade percentual.
Organização de Imagens Desconhecidas: Se o rosto não for identificado, a imagem é automaticamente salva na pasta unknown para verificação posterior.
Atualização de Embeddings: O sistema permite atualizar automaticamente os embeddings de cada aluno caso novas fotos sejam adicionadas, garantindo um reconhecimento mais preciso ao longo do tempo.

## 🧪 Modelos Utilizados

Para o reconhecimento facial, o projeto utiliza o modelo buffalo_l do InsightFace, com alto desempenho em precisão de reconhecimento em múltiplas condições de iluminação e ângulos.

Configuração de Modelos:
Detecção: SCRFD-10GF para detecção facial de alta precisão.
Reconhecimento: ResNet50, treinado no WebFace600K, para gerar embeddings faciais.
Atributos: Suporte para reconhecimento de idade e gênero.
Threshold de Similaridade: Ajustado para otimizar a precisão de reconhecimento com um limite mínimo de 0.65.

## ✒️ Autores

Este projeto foi desenvolvido como parte das atividades da disciplina de IOT pelos seguintes alunos:

* **Marcos Edson Anerio Dos Santos** - *Desenvolvedor* - [Marcos Edson](https://github.com/MarcosEdsonAnerio)
* **Wester Jesuino Morandi de Oliveira** - *Desenvolvedor* - [Wester Jesuino](https://github.com/MarcosEdsonAnerio)
* **Danilo Saiter da Silva** - *Desenvolvedor* - [Danilo Saiter](https://github.com/MarcosEdsonAnerio)
* **Ádrian Henrique Ferreira** - *Desenvolvedor* - [Ádrian Henrique](https://github.com/MarcosEdsonAnerio)

Professor responsável:

* **Wanderson Roger Azevedo Dias** - [Wanderson Roger]()

---
