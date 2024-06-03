from mtool.meval import *
from mtool.mio import get_medical_image
import os

# dice hf ....
# gt_path = './data/single_pulp/pulp/'
# gen_path = './result/pulp/'
#
# with os.scandir(gt_path) as files:
#     for file in files:
#         gt_tooth = get_medical_image(os.path.join(gt_path, file.name))[0]
#         gen_tooth = get_medical_image(os.path.join(gen_path, file.name))[0]
#         dice = get_dice(gen_tooth, gt_tooth)
#         print(file.name)
#         print(dice)

gt_path = './data/single/tooth/HU XIN YU-14.nrrd'
gen_path = './result1/tooth/HU XIN YU-14.nrrd'

gt_tooth = get_medical_image(gt_path)[0]
gen_tooth = get_medical_image(gen_path)[0]

dice = get_dice(gt_tooth, gen_tooth)
print(dice)