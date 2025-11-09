import os

class Utils:
    @staticmethod
    def load_documents():
        folder_path = 'documents'
        text_data = {}
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_data[filename.replace(".txt","")] = file.read()
        return text_data
