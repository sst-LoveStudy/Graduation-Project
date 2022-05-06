class VineVideo:
    def __init__(self, vid:str, frames_num:int, frames_list):
        self.vid = vid
        self.frames_num = frames_num
        self.frames_list = frames_list  # 存储的是每一帧的路径

    def display(self):
        print('This video:\n',
              'vid: ', self.vid,
              'frames_num: ', self.frames_num)