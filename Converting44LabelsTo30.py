
import glob
from scipy import io

# emiting holophens
# convertinfg silence to 30
# emiting labels of closues

PathLabels="D:\Shapar\ShaghayeghUni\AfterPropozal\MyPrograms\EventExtraction\SmallFarsdat\LABEL"

Labels30=[]
def Converting44LabelsTo30():
    LabelFiles=glob.glob(PathLabels+"/*.mat")
    for i,j in enumerate(Files):
        LabelFilePath=
        Labels=io.loadmat(LabelFilePath+".mat")
        LenFile=len(Labels)
        for k in range (LenFile):
            if Labels(k)==30:  Labels(k)=28
            if Labels(k)==31:  Labels(k)=28
            if Labels(k)==32:  Labels(k)=28
            if Labels(k)==33:  Labels(k)=28
            if Labels(k)==34:  Labels(k)=28
            if Labels(k)==35:  Labels(k)=28
            if Labels(k)==36:  Labels(k)=28
            if Labels(k)==37:  Labels(k)=28
            if Labels(k)==38:  Labels(k)=28
            if Labels(k)==39:  Labels(k)=28
            if Labels(k)==40:  Labels(k)=28
        Labels30.append(Labels)
    save(Labels30)
    return


