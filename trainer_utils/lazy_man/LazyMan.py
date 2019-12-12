from PIL import Image

# CALTECH LABELME PASCAL SUN
class LazyMan:
    def __init__(self, domain_list, target_domain_list):
        self.domain_list = domain_list
        self.target_domain_list = target_domain_list
        # self.number_of_domains = len(domain_list)
        self.source_and_target_domain_permutation_list = \
        self._set_source_and_target_domain_permutation_list()

    def _set_source_and_target_domain_permutation_list(self):
        source_and_target_domain_permutation_list = []
        for target_domain in self.target_domain_list:
            source_domain = []
            for i in self.domain_list:
                if i != target_domain:
                    source_domain.append(i)
            domain_dictionary = {
                'target_domain': target_domain,
                'source_domain':source_domain
            }
            source_and_target_domain_permutation_list.append(domain_dictionary)
        return source_and_target_domain_permutation_list

#
# lazy_man = LazyMan(['CALTECH', 'LABELME', 'PASCAL', 'SUN'])
# for item in lazy_man.source_and_target_domain_permutation_list:
#     print(item['target_domain'], item['source_domain'])


##


img = Image.open('/home/giorgio/Files/pycharm_project/fmc/data/VLCS/SUN/crossval/0/crossval_imgs_1.jpg')
img.show()

