

class Constant:
    def __init__(self):
        self.root_dir = "C:/Users/floyd/Desktop/test/financial-big-data-project/" # Change this to your root directory
        self.data_dir = self.root_dir + 'data/'
        self.raw_data_dir = self.data_dir + 'raw/'
        self.clean_data_dir = self.data_dir + 'clean/'
        self.processed_data_dir = self.data_dir + 'processed/'
        self.API_KEY = "YOUR_API_KEY" # Change this to your API key
        self.API_SECRET = "YOUR_API_SECRET" # Change this to your API secret
        self.FNSPID_data_path = self.raw_data_dir + "nasdaq_exteral_data.csv" 

if __name__ == "__main__":
    pass