import cv2
import os
from update_posProcessing import Final_Binary_convert
from predictor import Predictor
import numpy as np
import json
import time
class OCR():
    def __init__(self):
        self.__predictor = Predictor()
    def __crop_img(self, img, bounding_box):
        x, y, w, h, _ = bounding_box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        crop_img = img[y:y+h, x:x+w]
        ret, img_threshold = cv2.threshold(crop_img, 220, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        top, bot, left, right = 0, 0, 0, 0
        if w > h:
            delta = w - h
            top += delta//2
            bot += delta//2
        if w < h:
            delta = h - w
            left += delta//2
            right += delta//2
        top += 10
        bot += 10
        left += 10
        right += 10
        crop_img = cv2.copyMakeBorder(img_threshold, top, bot, left, right, cv2.BORDER_CONSTANT, (0, 0, 0))
        res = cv2.resize(crop_img, (50, 50))
        return np.expand_dims(res, 2)


    def __split_text_lines(self, img):
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else: 
            gray = img
        th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        hist = cv2.reduce(threshed,1, cv2.REDUCE_AVG).reshape(-1)

        th = 2
        H,W = img.shape[:2]
        uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
        uppers = uppers[1:]
        res = []
        for idx in range(len(uppers)):
            if idx == 0:
                res.append((0, 0, W, uppers[idx]))
            elif idx < len(uppers)-1:
                res.append((0, uppers[idx-1]+1, W, uppers[idx]))
        res.append((0, uppers[len(uppers)-1]+1, W, H))
        return res

    def __std_for_showing(self, result):
        lst_text = []
        lst_images = []
        for field in result:
            field_img, field_characters_img, field_result = result[field]
            lst_text.append(field)
            
            lst_images.append(field_img)
            for i in range(len(field_characters_img)):
                img, pred_char = field_characters_img[i], field_result[i]
                tmp_img = cv2.cvtColor(np.squeeze(img), cv2.COLOR_GRAY2BGR)
                # path = "collect_data/{}_{}.jpg".format(pred_char, time.time())
                # saved_img = cv2.resize(tmp_img, (50, 50))
                # cv2.imwrite(path, saved_img)
                tmp_img = cv2.resize(tmp_img, (70, 70))
                lst_images.append(tmp_img)
                lst_text.append("{} - {}%".format(pred_char[0], int(round(pred_char[1]*100))))
        return (lst_text, lst_images)


    def __std_for_serving(self, result):
        ocr_dict = {}
        field_img, field_characters_img, field_result = result["issue_office_0"]
        line = ""
        for i in range(len(field_characters_img)):
            pred_char = field_result[i]
            line += str(pred_char[0])
        ocr_dict["issue_office"] = line + " - "
        field_img, field_characters_img, field_result = result["issue_office_1"]
        for i in range(len(field_characters_img)):
            pred_char = field_result[i]
            line += str(pred_char[0])
        ocr_dict["issue_office"] = ocr_dict["issue_office"] + line
        for field in result:
            if ("issue_office" in field) or ("allowed_types" in field): continue
            field_img, field_characters_img, field_result = result[field]
            line = ""
            for i in range(len(field_characters_img)):
                pred_char = field_result[i]
                line += str(pred_char[0])
            if field == "issue_date_and_inquiry_number":
                ocr_dict["issue_date"] = line[:-5]
                ocr_dict["inquiry_number"] = line[-5:]
            elif field != "allowed_types":
                ocr_dict[field] = line
        return json.dumps(ocr_dict)


    def __process_muliple_lines(self, field_name, image):
        lines = self.__split_text_lines(image)
        res = {}
        for idx, l in enumerate(lines):
            x, y, w, h = l
            print(x, y, w, h)
            field_img = image[y:y+h, x:x+w]
            if field_name == "issue_office":
                path = "collect_data/{}_{}.jpg".format(field_name, time.time())
                # saved_img = cv2.resize(tmp_img, (50, 50))
                cv2.imwrite(path, field_img)
            _, stats_all = Final_Binary_convert(field_img)
            field_characters_img = []
            for i, st in enumerate(stats_all):
                croped_img = self.__crop_img(field_img, st)
                if i == 0:
                    field_characters_img = np.expand_dims(croped_img, axis=0)
                else:
                    field_characters_img = np.concatenate((field_characters_img, np.expand_dims(croped_img, axis=0)))
            if len(field_characters_img) == 0:
                continue
            field_result = None
            if field_name != "inquiry_number" and field_name != "license_number":
                field_result = self.__predictor.predict_all_character(input_images=field_characters_img)
            else:
                field_result = self.__predictor.predict_all_character(input_images=field_characters_img)
            key = "{}_{}".format(field_name, idx)
            res[key] = (field_img, field_characters_img, field_result)
        return res

    def __process_single_lines(self, field_name, image):
        _, stats_all = Final_Binary_convert(image)
        field_characters_img = []
        for i, st in enumerate(stats_all):
            croped_img = self.__crop_img(image, st)
            if i == 0:
                field_characters_img = np.expand_dims(croped_img, axis=0)
            else:
                field_characters_img = np.concatenate((field_characters_img, np.expand_dims(croped_img, axis=0)))
        if len(field_characters_img) == 0:
            return None
        field_result = None
        if field_name != "inquiry_number" and field_name != "license_number":
            field_result = self.__predictor.predict_all_character(input_images=field_characters_img)
        else:
            field_result = self.__predictor.predict_all_character(input_images=field_characters_img)
        return {field_name : (image, field_characters_img, field_result)}

    def __crop_field(self, img):
        # Select ROI
        r = None
        h,w,c = img.shape
        scale = 1
        print("fsdfsdfs",h,w,c)
        if h >= 720 and h < 2000 and w >= 1000 and w < 3000:
            print("1")
            tmp_img = img.copy()
            tmp_img = cv2.resize(tmp_img, (int(w/2), int(h/2)))
            scale = 2
            r = cv2.selectROI(tmp_img)
        elif h >= 2000 and w >= 3000:
            print("2")
            tmp_img = img.copy()
            tmp_img = cv2.resize(tmp_img, (int(w/4), int(h/4)))
            scale = 4
            r = cv2.selectROI(tmp_img)
        else:
            print("3")
            r = cv2.selectROI(img)
        print(r)
        # Crop image
        imCrop = img[int(r[1]*scale):int(r[1]*scale+r[3]*scale), int(r[0]*scale):int(r[0]*scale+r[2]*scale)]
        # cv2.imshow("tmp", imCrop)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        return imCrop, {"croped_image":[0,0,int(r[2]*scale),int(r[3]*scale)]}

    def OCR_driver_license(self, img, std_for_serving = False, std_for_debug = False):
        assert not (std_for_debug and std_for_serving), ("Not support for both mode debug and serving") 
        if img.ndim == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        processed_img, license_info = self.__crop_field(img)
        # return processed_img, license_info
        img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        result = {}
        for key in license_info:
            try:
                if key != "allowed_types":
                    bounding_box = license_info[key]
                    x, y, x1, y1 = bounding_box
                    x = int(x)
                    y = int(y)
                    w = int(x1-x)
                    h = int(y1-y)
                    field_img = img[y:y+h, x:x+w]
                    # path = "collect_data/{}_{}.jpg".format(key, time.time())
                    # # saved_img = cv2.resize(tmp_img, (50, 50))
                    # cv2.imwrite(path, field_img)
                    if key != "issue_office":
                        r = self.__process_single_lines(key, field_img)
                        if r is not None:
                            result[key] = r[key]
                    else:
                        info_dict = self.__process_muliple_lines(key, field_img)
                        for line in info_dict:
                            result[line] = info_dict[line]
                else:
                    for idx, bounding_box in enumerate(license_info[key]):
                        x, y, x1, y1 = bounding_box
                        x = int(x)
                        y = int(y)
                        w = int(x1-x)
                        h = int(y1-y)
                        field_img = img[y:y+h, x:x+w]
                        # path = "collect_data/{}_{}.jpg".format(key, time.time())
                        # # saved_img = cv2.resize(tmp_img, (50, 50))
                        # cv2.imwrite(path, field_img)
                        info_dict = self.__process_muliple_lines("{}_{}".format("allowed_types", idx), field_img)
                        for line in info_dict:
                            result[line] = info_dict[line]
            except:
                print(key)
                exit(-1)
        if std_for_debug:
            return self.__std_for_showing(result)
        if std_for_serving:
            return self.__std_for_serving(result)
        return result