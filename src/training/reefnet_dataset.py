"""
    Inspired by BioClip data loader 
    check original code of BioClip: "src/training/data.py" the class CsvDataset
    the function to_classes from ..imageomics.naming_eval is being used here 
    to create the taxonomic tree from Kingdom to genus.
"""
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from ..imageomics.naming_eval import to_classes
import torch
import open_clip
import numpy as np

def img_loader(filepath):
    img = Image.open(filepath)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    return img

# Define a function to extract the second word 
# this function can be used when we finetube bioclip to the species level
def get_second_word(x):
    if pd.isna(x) or not x.strip():  # Check for NaN or empty string
        return ''
    words = x.split()
    return words[1] if len(words) > 1 else ''


# Define a custom dataset class to load images
class FineTuneDataset(Dataset):
    def __init__(self, root_dir='/ibex/project/c2253/CoralNet_Images/', 
    csv_file='annotations_with_aphiaid_and_taxonomy.csv', column_name='taxon', transform=None,
    split='train', debug_loader=False, random_seed=1234, patch_type='224'):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.column_name = column_name
        # use the original tokenizer from BioCLIP
        self.tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file), low_memory=False).fillna('') #pd.read_csv(csv_file, index_col=0, low_memory=False).fillna('')
        if patch_type == '512':
            self.data  = self.data[self.data['Category'] == 'Hard corals']
            self.data  = self.data[(self.data['Taxonomic_Rank'] == 'Genus') | (self.data['Taxonomic_Rank'] == 'Species')]
        else:
            self.data  = self.data[(self.data['Taxonomic_Rank'] == 'Genus') | (self.data['Taxonomic_Rank'] == 'Species')]
            self.data  = self.data[(self.data['class'] == 'Hexacorallia') & (self.data['order'] == 'Scleractinia')]

        self.data = self.data.rename(columns={'class': 'cls'})
        list_of_sources = np.unique(self.data['Source'])

        # drop species column for genus global experiment
        self.data = self.data.drop(columns=['species'])
        # self.data['species'] = self.data['species'].apply(get_second_word)
        # Optional: take only a sample of the data
        
        # create the taxonomic label 
        self.data[self.column_name] = to_classes(self.data,self.column_name) # self.data[self.column_name].apply(lambda x: to_prompt(x)) # 
        # self.data['prompt'] = self.data[self.column_name].apply(lambda x: to_prompt(x))
        self.classes = self.data[self.column_name].unique()
        empty_classes = self.data[self.data[self.column_name] == '']
        self.class_to_idx = dict(zip(self.classes,range(len(self.classes))))
        # print(f'number of classes found: {len(self.classes)}')
        # print(f'classes found : {self.classes}')
        # print(self.class_to_idx)
        self.data['class_idx'] = self.data[self.column_name].apply(lambda x: self.class_to_idx[x])
        self.idx_to_class = dict([(v,k)for k,v in self.class_to_idx.items()])

        # test sources
        tropical_atlantic = ["curacao","STINAPA GCRMN"]
        easter_indo_pacific = ["CREP-REA SAMOA_PRIA v2",]
        central_indo_pacific = ["Tonga_2022-08 & Samoa_2022-09 & Samoa_2019-12 & Samoa_2017-12", "Penida benthic surveys", "WAPA Coral Inventory 2.0",]
        west_indo_pacific = ["Southern Arabian Gulf Biodiversity Assessment 2019", "REEFolution Kenya", "Maldives_Katie"]
        
        test_sources_list = tropical_atlantic + easter_indo_pacific + central_indo_pacific + west_indo_pacific
        train_sources_list = [src for src in list_of_sources if src not in test_sources_list]

        # source selection for train and test
        all_annotations_train = self.data[self.data['Source'].isin(train_sources_list)]
        all_annotations_test = self.data[self.data['Source'].isin(test_sources_list)]
        
        print(f"> Annotations in train/test annotations  {self.column_name}: {len(all_annotations_train)} / {len(all_annotations_test)}")
        # print(f"> [BEFORE FILTERING] Number of classes in train/test annotations {len(train_classes)}/{len(test_classes)}")

        # filtering out classes that have <2 annotations -> Updated train_classes and annotations
        unique_classes, samples_per_class = np.unique(all_annotations_train[self.column_name], return_counts=True)
        train_classes = unique_classes[samples_per_class > 1]
        all_annotations_train = all_annotations_train[all_annotations_train[self.column_name].isin(train_classes)]


        test_classes = list(np.unique(all_annotations_test[self.column_name]))
        test_not_in_train_classes = [class_i for class_i in test_classes if class_i not in train_classes]
        train_not_in_test_classes = [class_i for class_i in train_classes if class_i not in test_classes]


        all_annotations_train = all_annotations_train.reset_index(drop=True)
        all_annotations_test = all_annotations_test[all_annotations_test[self.column_name].isin(train_classes)].reset_index(drop=True)
        test_classes = list(np.unique(all_annotations_test[self.column_name]))

        self.test_classes = test_classes

            
        print(f"> Train/Test classes: {len(train_classes)} / {len(test_classes)}")
        print(f"> Test classes not in train: {len(test_not_in_train_classes)}/{len(test_classes)}")
        print(f" That are: {test_not_in_train_classes}")
        print(f"> Train classes not in test: {len(train_not_in_test_classes)}/{len(train_classes)}")
        print(f" That are: {train_not_in_test_classes}")
        print("> Train classes:")
        print(train_classes)
        print("> Test classes:")
        print(test_classes)
        print("---"*5)

        self.all_categories = train_classes
        self.labels_projection = {v: i for i, v in enumerate(train_classes)}

        if split == 'train':
            self.merged_annotations = all_annotations_train[:1000] if debug_loader else all_annotations_train
        elif split == 'val':
            self.merged_annotations = all_annotations_test[:1000] if debug_loader else all_annotations_test.sample(frac=0.25, random_state=random_seed).reset_index(drop=True)
        else:
            print("Chosing test split")
            self.merged_annotations = all_annotations_test[:1000] if debug_loader else all_annotations_test

        print(f">> Total {split} annotations annotations: {len(self.merged_annotations)}")

        unique, samples_per_class = np.unique(all_annotations_train[self.column_name], return_counts=True)
        self.samples_per_class_log = np.log(samples_per_class)
        print('Dataset class initialization done')
    
    def __len__(self):
        return len(self.merged_annotations)
    
    def __getitem__(self, idx):
        item = self.merged_annotations.iloc[idx]
        filepath = item['patch_path'] # os.path.join(self.root_dir, item['patch_path'])
        img = img_loader(filepath)
        if img is None:
            return None, None  # Handle image loading failure gracefully
        
        if self.transform is not None:
            img = self.transform(img)

        # Tokenize the text dynamically using the BioClip tokenizer
        text_tokens = self.tokenizer([str(item[self.column_name])])[0]  # Matching how BioClip does tokenization
        return img, text_tokens.long()


if __name__ == '__main__':
    main_path = '/ibex/project/c2253/CoralNet_Images/512_patches_with_aphiaids/'
    device = torch.device('cuda')
    # Initialize the dataset and dataloader
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip',device=device)
    dataset = FineTuneDataset(root_dir=main_path, transform=preprocess_train, classes='taxon', split='train', debug_loader=False)
    
    # (self, root_dir = '/ibex/project/c2253/CoralNet_Images/512_patches_with_aphiaids', 
    # csv_file='all_reefnet_annotations_with_taxonomy.csv', column_name='taxon', transform=None,classes='taxon',
    # split='train', debug_loader=False)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Generate zero-shot classifier for species
    classes = dataset.classes