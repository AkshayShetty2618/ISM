
''' The program gives you the automatic color equilization which is Algorithm for digital images unsupervised 
    enhancement based on a computational approach that merges the “gray world” and “white patch” equalization mechanisms 
    As prerequistes
    pip install 'colorcorrect' , 'numpy' and 'Pillow' '''



from PIL import Image
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil

#Add any skin cancer image here

img = Image.open('D:\ISM\python code\image2.jpg')
out= to_pil(cca.automatic_color_equalization(from_pil(img),10,1000)).show()

# for now manual save this image for further process
 
'''save this file to utilize it for HSV segmentation'''

#out.save(out.jpg)
#automatic_color_equalization()