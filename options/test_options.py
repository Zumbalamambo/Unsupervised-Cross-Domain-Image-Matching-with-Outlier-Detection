import argparse

class TrainOptions():
    def __init__(self):
        self.initialized = argparse.ArgumentParser()

    def initialize(self):
        parser = self.initialized
        parser.add_argumnet('--databaseSetPath', type=str, default='pits_database_path.txt', help='Database data name path for evalutation')
        parser.add_argumnet('--databaseImagePath', type=str, default='/tudelft.net/staff-bulk/ewi/insy/VisionLab/xinliu/pitsn/', help='Database data path')
        parser.add_argumnet('--querySetPath', type=str, default='queryPitsOut.txt', help='Query data name path for evalutation')
        parser.add_argumnet('--queryImagePath', type=str, default='/home/nfs/xliu7/CycleGAN-tensorflow/test/', help='Query data path')
        parser.add_argumnet('--checkpointModel', type=str, default='/tudelft.net/staff-bulk/ewi/insy/VisionLab/xinliu/checkpoints_weightedPits/model_epoch', help='Path to the model in checkpoints to be tested on')

        return parser
