import argparse

class TrainOptions():
    def __init__(self):
        self.initialized = argparse.ArgumentParser()

    def initialize(self):
        parser = self.initialized
        parser.add_argumnet('--train_DomainS_path', type=str, default='pits_pathS.txt', help='Source domain training data name path')
        parser.add_argumnet('--train_DomainT_path', type=str, default='imgTargetPS.txt', help='Target domain training data name path')
        parser.add_argumnet('--checkpointPath', type=str, default='/tudelft.net/staff-bulk/ewi/insy/VisionLab/xinliu/checkpoints_weightedPits', help='Path to store the checkpoints')
        parser.add_argumnet('--tr_DS_data', type=str, default='/tudelft.net/staff-bulk/ewi/insy/VisionLab/xinliu/pitsn/', help='Path to source domain training data')
        parser.add_argumnet('--tr_DT_data', type=str, default='/tudelft.net/staff-bulk/ewi/insy/VisionLab/xinliu/checkpoints_weightedPits', help='Path to target domain training data')

        return parser
