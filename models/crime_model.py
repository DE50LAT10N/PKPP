import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.cluster import OPTICS
import numpy as np
from typing import List, Dict
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

class CrimeDataset:
    def __init__(self, texts_df, tokenizer, text_column=None, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if not isinstance(texts_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if texts_df.empty:
            raise ValueError("DataFrame is empty")
            
        if text_column is None:
            text_column = texts_df.columns[0]
            
        if text_column not in texts_df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
            
        if texts_df[text_column].isna().all() or texts_df[text_column].str.strip().eq('').all():
            raise ValueError(f"Column '{text_column}' contains only empty values")
            
        self.texts = texts_df[text_column].values.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class CrimeModel:
    def __init__(self):
        self.texts = []
        self.cluster_labels = []
        self.clusters = []
        self.embeddings = None
        self.generated_texts = {}
        self.dataset = None  
        
    def load_data(self, file_path: str) -> Dict:
        texts_about_crimes = pd.read_excel(file_path)
        
        tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
        model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')
        
        self.dataset = CrimeDataset(texts_about_crimes, tokenizer)
        self.texts = self.dataset.texts
        
        self.embeddings = self._get_embeddings(self.texts, model, tokenizer)
        
        cluster_labels, valid_clusters = self._cluster_texts_with_optics()
        
        self.cluster_labels = cluster_labels
        self.clusters = valid_clusters
        
        return {
            'texts': self.texts,
            'embeddings': self.embeddings,
            'cluster_labels': cluster_labels,
            'clusters': valid_clusters
        }
    
    def _get_embeddings(self, texts, model=None, tokenizer=None, batch_size=32):
        model_name = "DeepPavlov/rubert-base-cased"
        
        if model is None:
            config = AutoConfig.from_pretrained(
                model_name,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=True
            )
            model = AutoModel.from_pretrained(
                model_name,
                config=config,
                add_pooling_layer=False
            )
            model.eval()
        
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if texts is None:
            texts = [""]
        elif isinstance(texts, str):
            texts = [texts]
        elif not texts:
            return torch.zeros((0, 768)) 
        
        texts = [str(text) if text is not None else "" for text in texts]
        
        if len(texts) == 1:
            encoding = tokenizer.encode_plus(
                texts[0],
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = model(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask']
                )
                return outputs.last_hidden_state[:, 0, :]
        
        else:
            embeddings = []
            
            dataset = CrimeDataset(pd.DataFrame({'text': texts}), tokenizer)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(model.device)
                    attention_mask = batch['attention_mask'].to(model.device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    last_hidden_states = outputs.last_hidden_state
                    cls_embeddings = last_hidden_states[:, 0, :]
                    embeddings.append(cls_embeddings)
            
            return torch.cat(embeddings, dim=0)
    
    def _cluster_texts_with_optics(self, embeddings=None, min_samples=2, xi=0.01, min_cluster_size=3):
        if embeddings is None:
            if self.embeddings is None:
                raise ValueError("No embeddings provided and no existing embeddings found")
            embeddings = self.embeddings
            
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        embeddings = embeddings.astype(np.float32)

        optics_model = OPTICS(
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            n_jobs=-1
        )
        
        optics_model.fit(embeddings)
        
        labels = optics_model.labels_
        valid_clusters = []
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        valid_mask = (counts >= min_cluster_size) & (unique_labels != -1)
        valid_clusters = list(zip(unique_labels[valid_mask], counts[valid_mask]))
        
        return labels, valid_clusters
    
    def get_formatted_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
            
        entities = self._extract_entities(text, ['LOC', 'PER', 'PRODUCT'])
        weapons = self._extract_weapons(text)
        
        formatted_text = text
        
        metadata = []
        
        if weapons:
            metadata.append(f"Оружие: {', '.join(weapons)}")
        if entities.get('LOC'):
            metadata.append(f"Места: {', '.join(entities['LOC'])}")
        if entities.get('PER'):
            metadata.append(f"Участники: {', '.join(entities['PER'])}")
            
        if metadata:
            formatted_text += "\n\n" + "\n".join(metadata)
            
        return formatted_text
    
    def get_cluster_texts(self, cluster_label: int) -> List[str]:
        texts = [self.texts[i] for i in range(len(self.texts))
                if self.cluster_labels[i] == cluster_label]
        return [self.get_formatted_text(text) for text in texts]
    
    def add_generated_text(self, cluster_label: int, text: str):
        if cluster_label not in self.generated_texts:
            self.generated_texts[cluster_label] = []
        self.generated_texts[cluster_label].append(text)
    
    def get_generated_texts(self, cluster_label: int) -> List[str]:
        texts = self.generated_texts.get(cluster_label, [])
        return [self.get_formatted_text(text) for text in texts]
    
    def export_to_excel(self, file_path: str):
        if not self.texts or not self.clusters:
            raise ValueError("No data to export")

        excel_data = []
        
        for cluster_label, cluster_size in self.clusters:
            cluster_texts = self.get_cluster_texts(cluster_label)
            
            for text in cluster_texts:
                entities = self._extract_entities(text, ['LOC', 'PER', 'PRODUCT'])
                weapons = self._extract_weapons(text)
                
                excel_data.append({
                    'Cluster': f"Cluster {cluster_label} (Original)",
                    'Text': text,
                    'Weapons': ', '.join(weapons) if weapons else 'No weapons found',
                    'Locations': ', '.join(entities.get('LOC', [])) if entities.get('LOC') else 'No locations found',
                    'Participants': ', '.join(entities.get('PER', [])) if entities.get('PER') else 'No participants found'
                })
            
            if cluster_label in self.generated_texts:
                for text in self.generated_texts[cluster_label]:
                    entities = self._extract_entities(text, ['LOC', 'PER', 'PRODUCT'])
                    weapons = self._extract_weapons(text)
                    
                    excel_data.append({
                        'Cluster': f"Cluster {cluster_label} (Generated)",
                        'Text': text,
                        'Weapons': ', '.join(weapons) if weapons else 'No weapons found',
                        'Locations': ', '.join(entities.get('LOC', [])) if entities.get('LOC') else 'No locations found',
                        'Participants': ', '.join(entities.get('PER', [])) if entities.get('PER') else 'No participants found'
                    })
        
        if not excel_data:
            raise ValueError("No data to export")
        
        df = pd.DataFrame(excel_data)
        if 'Cluster' in df.columns:
            df['Cluster_Num'] = df['Cluster'].str.extract(r'Cluster (\d+)').astype(int)
            df = df.sort_values(['Cluster_Num', 'Cluster'])
            df = df.drop('Cluster_Num', axis=1)
        
        df.to_excel(file_path, index=False)
    
    def _extract_entities(self, text: str, entity_types: List[str] = None) -> Dict[str, List[str]]:
        if not text or not isinstance(text, str):
            return {entity_type: [] for entity_type in entity_types} if entity_types else {}
        
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        doc.tag_ner(ner_tagger)
        
        result = {entity_type: [] for entity_type in entity_types} if entity_types else {}
        
        for span in doc.spans:
            if entity_types is None or span.type in entity_types:
                if span.type not in result:
                    result[span.type] = []
                if span.text not in result[span.type]:
                    result[span.type].append(span.text)
        
        return result
    
    def _extract_weapons(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        
        weapons = []
        text_lower = text.lower()

        weapon_keywords = {
            'нож', 'пистолет', 'ружье', 'ружьё', 'автомат', 'револьвер', 
            'винтовка', 'обрез', 'кастет', 'дубинка', 'бита', 'топор',
            'лезвие', 'клинок', 'оружие', 'ствол', 'арматура', 'молоток',
            'палка', 'холодное оружие', 'огнестрельное оружие'
        }
        plural_map = {
            'ножи': 'нож', 'пистолеты': 'пистолет', 'ружья': 'ружье', 'ружья': 'ружьё',
            'автоматы': 'автомат', 'револьверы': 'револьвер', 'винтовки': 'винтовка',
            'обрезы': 'обрез', 'кастеты': 'кастет', 'дубинки': 'дубинка', 'биты': 'бита',
            'топоры': 'топор', 'лезвия': 'лезвие', 'клинки': 'клинок', 'стволы': 'ствол',
            'арматуры': 'арматура', 'молотки': 'молоток', 'палки': 'палка',
            'оружия': 'оружие'
        }
        def normalize(word):
            for plural, singular in plural_map.items():
                if word == plural:
                    return singular
            for suf in ['у', 'ю', 'е', 'а', 'ы', 'и', 'ой', 'ей', 'ом', 'ем', 'ам', 'ям', 'ах', 'ях']:
                if word.endswith(suf):
                    base = word[:-len(suf)]
                    if len(base) >= 3:
                        for k in weapon_keywords:
                            if k.startswith(base) and abs(len(k) - len(base)) <= 2:
                                return k
            return word
        negative_phrases = [
            'без применения оружия',
            'без оружия',
            'никакого оружия',
            'не было оружия',
            'отсутствие оружия'
        ]
        if any(neg in text_lower for neg in negative_phrases):
            weapon_keywords = weapon_keywords - {'оружие', 'холодное оружие', 'огнестрельное оружие'}
        entities = self._extract_entities(text, ['PRODUCT'])
        for product in entities.get('PRODUCT', []):
            product_lower = product.lower()
            for keyword in weapon_keywords:
                if keyword in product_lower and keyword not in weapons:
                    weapons.append(keyword)
            for plural, singular in plural_map.items():
                if plural in product_lower and singular not in weapons:
                    weapons.append(singular)
        words = text_lower.split()
        for word in words:
            word = word.strip('.,!?()[]{}":;')
            norm = normalize(word)
            if norm in weapon_keywords and norm not in weapons:
                weapons.append(norm)
        return weapons 