# This is a sample Python script.
import os


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# import os
# # file=os.open('temp.txt',os.O_RDONLY)
# # print(file.)
#
#
# file=open('temp4.txt','x')
# file.write('This file has now more gaali')
# file.close()
#
# file=open('temp4.txt','r')
# print(file.read())
#
# if(os.path.exists('temp.txt')):
#     os.remove('temp.txt')
# else:
#     file=open('temp.txt','x')
#     os.remove('temp.txt')
# import os
# print(os.getcwd())
# temp=os.getcwd()
# os.makedirs('BGU/data/temp_spectral4')
# os.chdir(os.path.join(os.getcwd(),'BGU/data'))
# os.mkdir('temp_clean4')
# print(os.getcwd())
# os.chdir(temp)
# print(os.listdir())
# os.rename('understanding os','Understanding OS')

# import shutil
# import os
# temp=os.getcwd()
# source_path=os.path.join(os.getcwd(),'BGU/data/temp_clean4')
# dest_path=os.path.join(os.getcwd(),'BGU2/data')
# shutil.move(source_path,dest_path)


############# This function takes your path from current dir , source path and dist path and how many files needed to transfer ##########


# import os
# import shutil
# def doit(source_path,dist_path,n):
#     temp=os.getcwd()
#     os.chdir(os.path.join(temp,source_path))
#     list=[]
#     for files in os.listdir():
#         list.append(files)
#     os.chdir(temp)
#     dist_path2=os.path.join(temp,dist_path)
#     index=0
#     for i in list:
#         if(index<n):
#             source_path=os.path.join(os.getcwd(),f'BGU/data/{i}')
#             shutil.move(source_path,dist_path2)
#             index=index+1
#         else:
#             return
#     return
#
# doit('BGU/data','BGU2/data',1)


################# This function moves 2 files together according to the conventions in the BGU_dataset ###############

import os
import shutil
def doit(old_spectral,old_clean,new_spectral,new_clean,n):
    temp=os.getcwd()
    list=[]
    os.chdir(os.path.join(temp,old_spectral))
    for files in os.listdir():
        list.append(files)
    os.chdir(temp)
    dist_path1=os.path.join(os.getcwd(),new_spectral)
    dist_path2=os.path.join(os.getcwd(),new_clean)
    index=0
    for i in list:
        if(index<n):
            source_path1=os.path.join(os.getcwd(),old_spectral,i)
            source_path2=os.path.join(os.getcwd(),old_clean,(i[:-4]+'_clean.png'))
            shutil.move(source_path1,dist_path1)
            shutil.move(source_path2,dist_path2)
            index=index+1
        else:
            return


doit('BGU/data/train_spectral','BGU/data/train_clean','BGU/data/val_spectral','BGU/data/val_clean',1)







