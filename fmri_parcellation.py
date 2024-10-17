import nibabel as nib
import numpy as np
import os

# 从AAL_90模板中找出每个脑区的mask
def get_roi_mask():
    # 使用先验脑区mask
    aal = nib.load('./aal.nii.gz')
    aal_data = aal.get_fdata()
    roi = np.empty([90, 181, 217, 181])
    # for i in range(181):
    #     for j in range(217):
    #         for k in range(181):
    #             da[i, j, k] = aal_data[2 * i, 2 * j, 2 * k]
    for i in range(90):
        roi[i] = np.where(aal_data == i + 1, 1, 0)

    roi_mask = np.empty([90, 64, 64, 48])
    for o in range(90):

        for i in range(64):
            i1 = int(i/64*181)

            for j in range(64):
                j1 = int(j/64*217)

                for k in range(48):
                    k1 = int(k/48*181)

                    roi_mask[o, i, j, k] = roi[o, i1, j1, k1]

    return roi_mask



if __name__ == '__main__':


    in_path = 'nc/020_S_6566_20180830_162631AxialrsfMRIEyesOpens011a001.nii.gz'
    tmp = nib.load(in_path)
    data = tmp.get_fdata() #  (64, 64, 48, 197) 
    print('===>  ',data.shape)

    roi_mask = get_roi_mask() #roi_mask = np.empty([90, 64, 64, 48])

    # 降采样  roi_mask = np.empty([90, 181, 217, 181])
    # roi_mask1 = np.empty([90, 64, 64, 48])
    # for o in range(90):

    #     for i in range(64):
    #         i1 = int(i/64*181)

    #         for j in range(64):
    #             j1 = int(j/64*217)

    #             for k in range(48):
    #                 k1 = int(k/48*181)

    #                 roi_mask1[o, i, j, k] = roi_mask[o, i1, j1, k1]

    timeseries=np.empty([90, 187])
    for i in range(197):
        if i<=9:
            continue
        for j in range(90):
            source = data[:,:,:,i]
            mask_j = roi_mask[j,:,:,:]
            extract_pixel = source*mask_j
            nonzero=(extract_pixel!=0)
            mean_value = extract_pixel.sum()/nonzero.sum()
            timeseries[j,i-10]=mean_value
            # print('roi=',j,'points=',i, ';  timeseries_value=',mean_value)

    timeseries_m = np.mean(timeseries,1)
    timeseriesT = timeseries.T - timeseries_m
    timeseries1 = timeseriesT.T
    print(timeseries1)
    maxvalue = np.max(abs(timeseries1))
    print("maxvalue= ",maxvalue)
    timeseries1 = timeseries1/maxvalue
    print(timeseries1)
    np.savetxt( in_path + '90ROI.txt', timeseries1, delimiter=' ')


