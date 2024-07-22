# yolov5源码修改
+ 静态导出修改export代码的输出名称output0为output
```
# yolov5-7.0/export.py第141行
# output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
# 修改为：

output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output']
```
+ 动态导出修改export的输出名称output0为output和yolo的前向传播函数定义动态输入
```
# yolov5-7.0/models/yolo.py第60行，forward函数
# bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
# x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
# 修改为：

bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
bs = -1
ny = int(ny)
nx = int(nx)
x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

# yolov5-7.0/export.py第141行
# output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
# if dynamic:
#     dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
#     if isinstance(model, SegmentationModel):
#         dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
#         dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
#         elif isinstance(model, DetectionModel):
#             dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
# 修改为：

output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output']            
if dynamic:
    dynamic = {'images': {0: 'batch'}}  # shape(1,3,640,640)
    if isinstance(model, SegmentationModel):
        dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
        dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    elif isinstance(model, DetectionModel):
        dynamic['output'] = {0: 'batch'}  # shape(1,25200,85)
```
# tensorRT_Pro第三方库修改
+ CMakeLists.txt修改python支持、opencv、cuda、cudnn、trt、protobuf路径，其中cudnn可以注释掉，包含在cuda里面了
+ workspace文件夹下准备一个calib校准集，1000张训练集图片即可
+ src/application/app_yolo.cpp修改用于导出trtmodel和执行推理
```
test(Yolo::Type::V5, TRT::Mode::INT8, "best")				// 修改1 177行"yolov7"改成"best"

static const char *voclabels[] = {"aeroplane",   "bicycle", "bird",   "boat",       "bottle",
                                  "bus",         "car",     "cat",    "chair",      "cow",
                                  "diningtable", "dog",     "horse",  "motorbike",  "person",
                                  "pottedplant",  "sheep",  "sofa",   "train",      "tvmonitor"};		 // 修改2 25行新增代码，为自训练模型的类别名称
    
for(auto& obj : boxes){
     ...
     auto name    = mylabels[obj.class_label];	 			// 修改3 100行cocolabels修改为mylabels
	 ...
}

TRT::compile(
    mode,                       // FP32、FP16、INT8
    test_batch_size,            // max batch size
    onnx_file,                  // source 
    model_file,                 // save to
    {},
    int8process,
    "calib_data"				// 修改4 149行 "inference" 修改为 "calib_data"
);

```
+ src/application/test_yolo_map.cpp修改用于计算map值
```
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/json.hpp>
#include "app_yolo/yolo.hpp"
#include <vector>
#include <string>

using namespace std;

bool requires(const char* name);

struct BoxLabel{
    int label;
    float cx, cy, width, height;
    float confidence;
};

struct ImageItem{
    string image_file;
    Yolo::BoxArray detections;
};

vector<ImageItem> scan_dataset(const string& images_root){

    vector<ImageItem> output;
    auto image_files = iLogger::find_files(images_root, "*.jpg");

    for(int i = 0; i < image_files.size(); ++i){
        auto& image_file = image_files[i];

        if(!iLogger::exists(image_file)){
            INFOW("Not found: %s", image_file.c_str());
            continue;
        }

        ImageItem item;
        item.image_file = image_file;
        output.emplace_back(item);
    }
    return output;
}

static void inference(vector<ImageItem>& images, int deviceid, const string& engine_file, TRT::Mode mode, Yolo::Type type, const string& model_name){

    auto engine = Yolo::create_infer(
        engine_file, type, deviceid, 0.001f, 0.65f,
        Yolo::NMSMethod::CPU, 10000
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    int nimages = images.size();
    vector<shared_future<Yolo::BoxArray>> image_results(nimages);
    for(int i = 0; i < nimages; ++i){
        if(i % 100 == 0){
            INFO("Commit %d / %d", i+1, nimages);
        }
        image_results[i] = engine->commit(cv::imread(images[i].image_file));
    }
    
    for(int i = 0; i < nimages; ++i)
        images[i].detections = image_results[i].get();
}

void detect_images(vector<ImageItem>& images, Yolo::Type type, TRT::Mode mode, const string& model){

    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            Yolo::image_to_tensor(image, tensor, type, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test %s %s %s ==================================", Yolo::type_name(type), mode_name, name);

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;
    
    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                 // save to
            {},
            int8process,
            "inference"
        );
    }
    inference(images, deviceid, model_file, mode, type, name);
}

bool save_to_json(const vector<ImageItem>& images, const string& file){

    Json::Value predictions(Json::arrayValue);
    for(int i = 0; i < images.size(); ++i){
        auto& image = images[i];
        auto file_name = iLogger::file_name(image.image_file, false);
        string image_id = file_name;

        auto& boxes = image.detections;
        for(auto& box : boxes){
            Json::Value jitem;
            jitem["image_id"] = image_id;
            jitem["category_id"] = box.class_label;
            jitem["score"] = box.confidence;

            auto& bbox = jitem["bbox"];
            bbox.append(box.left);
            bbox.append(box.top);
            bbox.append(box.right - box.left);
            bbox.append(box.bottom - box.top);
            predictions.append(jitem);
        }
    }
    return iLogger::save_file(file, predictions.toStyledString());
}

int test_yolo_map(){
    
    /*
    结论：
    1. YoloV5在tensorRT下和pytorch下，只要输入一样，输出的差距最大值是1e-3
    2. YoloV5-6.0的mAP，官方代码跑下来是mAP@.5:.95 = 0.367, mAP@.5 = 0.554，与官方声称的有差距
    3. 这里的tensorRT版本测试的精度为：mAP@.5:.95 = 0.357, mAP@.5 = 0.539，与pytorch结果有差距
    4. cv2.imread与cv::imread，在操作jpeg图像时，在我这里测试读出的图像值不同，最大差距有19。而png图像不会有这个问题
        若想完全一致，请用png图像
    5. 预处理部分，若采用letterbox的方式做预处理，由于tensorRT这里是固定640x640大小，测试采用letterbox并把多余部分
        设置为0. 其推理结果与pytorch相近，但是依旧有差别
    6. 采用warpAffine和letterbox两种方式的预处理结果，在mAP上没有太大变化（小数点后三位差）
    7. mAP差一个点的原因可能在固定分辨率这件事上，还有是pytorch实现的所有细节并非完全加入进来。这些细节可能有没有
        找到的部分
    */

    auto images = scan_dataset("/home/jarvis/Learn/Datasets/VOC_PTQ/images/val");
    INFO("images.size = %d", images.size());

    string model = "best";
    detect_images(images, Yolo::Type::V5, TRT::Mode::INT8, model);
    save_to_json(images, model + ".prediction.json");
    return 0;
}

```
+ 每次修改需要重新编译：
```
mkdir build && cd build
cmake ..&& make -j48
./pro yolo
./pro test_yolo_map
```
+ 用json文件计算map值所用代码
```
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Run COCO mAP evaluation
# Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

annotations_path = "/datasets/datasets/new_dataset/rename/annotations/calib_test_1.json"
results_file = "best.prediction.json"
cocoGt = COCO(annotation_file=annotations_path)
cocoDt = cocoGt.loadRes(results_file)
imgIds = sorted(cocoGt.getImgIds())
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

```
+ 校准集数量可以自行摸索,此前实验的结果如下

| 校准集大小  | map0.5-0.9 | map0.5 |
| ----------- | ---------- | ------ |
| 0           | 0.165      | 0.282  |
| 100         | 0.283      | 0.446  |
| 500         | 0.314      | 0.494  |
| 1000        | 0.28       | 0.446  |
| 2000        | 0.303      | 0.479  |
| 3000        | 0.311      | 0.491  |
| 5000        | 0.296      | 0.472  |
| 10000       | 0.312      | 0.492  |
| 118287(all) | 0.319      | 0.5    |
| 原模型      | 0.371      | 0.566  |


# 参考链接
https://blog.csdn.net/qq_40672115/article/details/133325424