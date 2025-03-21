# style_transfer

# 🏗️ Project Structure  

This repository is structured to efficiently handle **style transfer** using **StyleGAN** and **JoJoGAN**, incorporating data preprocessing, synthetic data generation, and training pipelines.  

## 📂 Directory Overview  

  ```plaintext
style_transfer/  
│  
├── data/                     
│   ├── __init__.py           
│   ├── notebooks/            
│   │   └── collected_data_eda.ipynb  
│   ├── preprocessing/     
│   │   ├── __init__.py       
│   │   ├── prepare_dataset.py  
│   │   ├── utils/           
│   │   │   ├── __init__.py  
│   │   │   ├── collection_utils.py  
│   │   │   ├── face_detector.py  
│   ├── scripts/             
│   │   ├── scrapping_lego_script.py  # LEGO-style image scraping  
│   │   ├── synth_gen.py      # Synthetic data generation  
│   │   ├── readme.md         
│   ├── utils/               
│   │   ├── __init__.py      
│   │   ├── image_dataset.py  
│  
├── scripts/                  
│   ├── stylegan/              
│   │   ├── training/         
│   │   │   ├── train.py      # Main training script  
│  
├── .gitignore               
├── README.md                
```

