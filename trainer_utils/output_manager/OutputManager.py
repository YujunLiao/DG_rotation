import numpy as np

class OutputManager:
    def __init__(self, output_file_path):
        self.output_file_path = output_file_path

    def write_to_output_file(self, str_list):
        with open(self.output_file_path, 'a') as f:
            for str in str_list:
                f.write(str+'\n')

    def read_from_output_file(self):
        with open(self.output_file_path, 'r') as f:
            return f.read()






# output_manager = OutputManager('/home/giorgio/Files/pycharm_project/DG_rotation/trainer_utils/output_manager/output_file/'+'12')
# for i in range(10):
#     order = np.random.randint(4)
#     print(order)
#     output_manager.write_to_output_file(str(i)+'dsfsdf')
#
# print(output_manager.read_from_output_file())
