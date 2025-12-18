import os.path
import json
import random

from torch.utils.data import Dataset
from skimage import io
from tqdm import tqdm
from PIL import Image
import numpy as np

class VQALoader(Dataset):
    """
    This class manages the Dataloading.
    """

    def __init__(self,
                 imgFolder,
                 images_file,
                 questions_file,
                 answers_file,
                 tokenizer,
                 image_processor,
                 Dataset,
                 train=True,
                 ratio_images_to_use=0.1,
                 selected_answers=None,
                 sequence_length=40,
                 transform=None,
                 label=None):

        self.train = train
        self.imgFolder = imgFolder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        LR_number_outputs = 9
        HR_number_outputs = 94
        self.Dataset = Dataset
        self.transform = transform
        self.dict = {}

        # sequence length of the tokens
        self.sequence_length = sequence_length

        print("Loading JSONs...")
        with open(questions_file) as json_data:
            questionsJSON = json.load(json_data)

        with open(answers_file) as json_data:
            answersJSON = json.load(json_data)

        with open(images_file) as json_data:
            imagesJSON = json.load(json_data)
        print("Done.")

        images = [img['id'] for img in imagesJSON['images'] if img['active']]

        self.img_ids = images
        if self.Dataset == 'LR':
            self.images = np.empty((len(images), 256, 256, 3))
        else:
            self.images = np.empty((len(images), 512, 512, 3))

        print("Construction of the Dataset")
        print("+++++++++++++++++++++++++++++++++++++train", train)
        if train:
            print("--------train----------------")
            self.freq_dict = {}
            all_anser = {}
            for i, image in enumerate(tqdm(images)):

                # select the questionids, aligned to the image
                for questionid in imagesJSON['images'][image]['questions_ids']:

                    # select question with the id
                    question = questionsJSON['questions'][questionid]

                    # get the answer str with the answer id from the question
                    answer_str = answersJSON['answers'][question["answers_ids"][0]]['answer']
                    if answer_str not in all_anser:
                        all_anser[answer_str] = 1
                    # group the counting answers
                    if self.Dataset == 'LR':
                        if answer_str.isdigit():
                            num = int(answer_str)
                            if num > 0 and num <= 10:
                                answer_str = "between 0 and 10"
                            if num > 10 and num <= 100:
                                answer_str = "between 10 and 100"
                            if num > 100 and num <= 1000:
                                answer_str = "between 100 and 1000"
                            if num > 1000:
                                answer_str = "more than 1000"
                    else:
                        if 'm2' in answer_str:
                            num = int(answer_str[:-2])
                            if num > 0 and num <= 10:
                                answer_str = "between 0m2 and 10m2"
                            if num > 10 and num <= 100:
                                answer_str = "between 10m2 and 100m2"
                            if num > 100 and num <= 1000:
                                answer_str = "between 100m2 and 1000m2"
                            if num > 1000:
                                answer_str = "more than 1000m2"

                    # update the dictionary
                    if answer_str not in self.freq_dict:
                        self.freq_dict[answer_str] = 1
                    else:
                        self.freq_dict[answer_str] += 1

            self.freq_dict = sorted(self.freq_dict.items(), key=lambda x: x[1], reverse=True)

            self.selected_answers = []
            self.non_selected_answers = []

            coverage = 0

            total_answers = 0

            for i, key in enumerate(self.freq_dict):

                if self.Dataset == 'LR':
                    if i < LR_number_outputs:

                        self.selected_answers.append(key[0])

                        coverage += key[1]

                    else:
                        self.non_selected_answers.append(key[0])

                    total_answers += key[1]
                else:
                    if i < HR_number_outputs:

                        self.selected_answers.append(key[0])

                        coverage += key[1]

                    else:
                        self.non_selected_answers.append(key[0])

                    total_answers += key[1]
            print(self.selected_answers)

        else:
            '''['yes', 'no', 'between 10 and 100', 'between 0 and 10', '0', 'between 100 and 1000', 'more than 1000', 'rural', 'urban']'''
            '''  main ['yes', 'no', 'between 0 and 10', '0', 'between 10 and 100', 'between 100 and 1000', 'more than 1000', 'rural', 'urban']'''
            print("-------------else ----")

            select_answer = ['yes', 'no', 'between 0 and 10', '0', 'between 10 and 100', 'between 100 and 1000',
                             'more than 1000', 'rural', 'urban']

            self.selected_answers = select_answer

        # list for storing the image-question-answer pairs
        self.images_questions_answers = []

        # we go through all img ids
        for i, image in enumerate(tqdm(images)):

            img = io.imread(os.path.join(imgFolder, str(image) + '.tif'))
            self.images[i, :, :, :] = img

            # use img id to get the question id for corresponding to the img
            for questionid in imagesJSON['images'][image]['questions_ids']:

                # question id gives the dict of the question
                question = questionsJSON['questions'][questionid]

                # get the question str and the question type (e.g. Y/N)
                question_str = question["question"]
                type_str = question["type"]

                # get the answer str with the answer id from the question
                answer_str = answersJSON['answers'][question["answers_ids"][0]]['answer']

                # group the counting answers
                if self.Dataset == 'LR':
                    if answer_str.isdigit():
                        num = int(answer_str)
                        if num > 0 and num <= 10:
                            answer_str = "between 0 and 10"
                        if num > 10 and num <= 100:
                            answer_str = "between 10 and 100"
                        if num > 100 and num <= 1000:
                            answer_str = "between 100 and 1000"
                        if num > 1000:
                            answer_str = "more than 1000"
                else:
                    if 'm2' in answer_str:
                        num = int(answer_str[:-2])
                        if num > 0 and num <= 10:
                            answer_str = "between 0m2 and 10m2"
                        if num > 10 and num <= 100:
                            answer_str = "between 10m2 and 100m2"
                        if num > 100 and num <= 1000:
                            answer_str = "between 100m2 and 1000m2"
                        if num > 1000:
                            answer_str = "more than 1000m2"

                answer = self.selected_answers.index(answer_str)


                if label is not None:
                    if type_str == label:
                        self.images_questions_answers.append([question_str, answer, i, type_str, answer_str])
                else:
                    self.images_questions_answers.append([question_str, answer, i, type_str, answer_str])

        print("Done.")

    def __len__(self):
      
        return len(self.images_questions_answers)



    def __getitem__(self, idx):

      
        data = self.images_questions_answers[idx]
      
        language_feats = self.tokenizer(
            data[0],
            return_tensors='pt',
            padding='max_length',
            max_length=self.sequence_length
        )
        # todo : 这里用来加提示

        new_question = data[0] + ' ' + data[3]

        tishi_language_feats = self.tokenizer(
            new_question,
            return_tensors='pt',
            padding='max_length',
            max_length=self.sequence_length

        )
     

        # 处理图像数据 w,h,c 256 256 3
        img = self.images[data[2], :, :, :]
      

        # todo:----------------------------------------------

        if self.train and data[1] in {2, 4, 5, 6}:
            if random.random() < .5:
                img = np.flip(img, axis=0)
            if random.random() < .5:
                img = np.flip(img, axis=1)
            if random.random() < .5:
                img = np.rot90(img, k=1)
            if random.random() < .5:
                img = np.rot90(img, k=3)

        if self.transform is not None:
            img_unaugment = Image.fromarray(np.uint8(img)).convert('RGB')
            img_augment = self.transform(img_unaugment)
            imgT = self.image_processor(img_augment, return_tensors="pt", do_resize=True)
        else:
            imgT = self.image_processor(img, return_tensors="pt")
    
        answer_idx = data[1]

    
        data4 = data[4]
        question_type_text = data[3]
     
        

        labels = self.tokenizer(data[4], padding='max_length', truncation=True, max_length=9, return_tensors="pt")
       
        label = labels['input_ids']
        label = label.squeeze(0)
        label_attention_mask = labels['attention_mask']

        if self.train:
            return {
                "pixel_values": imgT['pixel_values'][0], 
                "input_ids": language_feats['input_ids'][0],  
                "attention_mask": language_feats['attention_mask'][0],  
                "labels": label,
                "label_attention_mask": label_attention_mask,
                "question_type": data[3],  
                "answer": data[1],  
                "tishi_language_feats": tishi_language_feats['input_ids'][0],
                "tishi_attention_mask": tishi_language_feats['attention_mask'][0],
            }
        else:
      
            return {
                "pixel_values": imgT['pixel_values'][0], 
                "input_ids": language_feats['input_ids'][0],  
                "attention_mask": language_feats['attention_mask'][0],  
                "labels": label, 
                "image_id": data[2],  
                "question": data[0],  
                "answer_str": data[4],  
                "question_type": data[3], 
                "answer": data[1],  
                "label_attention_mask": label_attention_mask,
                "tishi_language_feats": tishi_language_feats['input_ids'][0],
                "tishi_attention_mask": tishi_language_feats['attention_mask'][0],

            }
