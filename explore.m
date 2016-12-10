%%  实验一
%{实现对图像文件的读/写/信息查询，图像显示--显示多幅图像、4种图像类型的显示方法、图像类型间的转换。  
C=imread('guku.jpg');
C=imresize(C,[320,480]);  
figure;
%RGB show
x=uint8(C);
subplot(2,2,1),imshow(x,[]);title('ＲＧＢ图像','FontSize',14);   
%Gray show
y=rgb2gray(x);
subplot(2,2,2),imshow(y,[]);title('灰度图像','FontSize',14);   
%转成二值图像 
H=uint8(y);
[Y,X,Z]=size(y);
for i=1:Y
    for j=1:X
        if H(i,j)<=100
        H(i,j)=0;
        else H(i,j)=255;
        end
    end
end
subplot(2,2,3),imshow(H,[]);title('二值图像','FontSize',14);  
%map show
[X,map]=gray2ind(C,200);
subplot(2,2,4),imshow(X,map);title('索引图像','FontSize',14);  
imwrite(X,'blackguku.jpg'); 
%%  实验二
%对图像文件分别进行平移、垂直镜像变换、水平镜像变换、缩放和旋转操作。

%% 平移
A=imread('guku.jpg'); 
A=imresize(A,[320,480]);  
A=uint8(A);
[a,b,c]=size(A);
B=ones(a+100,b+100,c);B=uint8(B);
for n=1:3
for i=1:a+100
for j=1:b+100
     if i<a&&j<b
         B(i,j,n)= A(i,j,n); 
     else
         B(i,j,n)= 255; 
     end
end
end
end
figure(1);imshow(B);title('origin');
Q=ones(a+100,b+100,c);Q=uint8(Q);

for n=1:3
for i=1:a+100
for j=1:b+100
     if j>80&&i<a&&j<b+80
         Q(i,j,n)= B(i,j-80,n); 
     else
         Q(i,j,n)= 255; 
     end
end
end
end
figure(2);imshow(Q);title('平移');
%% 垂直镜像变换
A=imread('guku.jpg'); 
A=imresize(A,[320,480]);  
A=uint8(A);
[a,b,c]=size(A);
B=ones(a+100,b+100,c);B=uint8(B);
for n=1:3
for i=1:a+100
for j=1:b+100
     if i<a&&j<b
         B(i,j,n)= A(i,j,n); 
     else
         B(i,j,n)= 255; 
     end
end
end
end
figure(1);imshow(B);title('origin');
Q=ones(a+100,b+100,c);Q=uint8(Q);
for n=1:3
for i=1:a+100
for j=1:b+100
          if i<a&&j<b
         Q(i,j,n)= B(i,1+b-j,n);  
          else
         Q(i,j,n)= 255; 
           end
end
end
end
figure(2);imshow(Q);title('垂直镜像变换');
%% 水平镜像变换
A=imread('guku.jpg'); 
A=imresize(A,[320,480]);  
A=uint8(A);
[a,b,c]=size(A);
B=ones(a+100,b+100,c);B=uint8(B);
for n=1:3
for i=1:a+100
for j=1:b+100
     if i<a&&j<b
         B(i,j,n)= A(i,j,n); 
     else
         B(i,j,n)= 255; 
     end
end
end
end
figure(1);imshow(B);title('origin');
Q=ones(a+100,b+100,c);Q=uint8(Q);
for n=1:3
for i=1:a+100
for j=1:b+100
          if i<a&&j<b
         Q(i,j,n)= B(a+1-i,j,n);  
          else
         Q(i,j,n)= 255; 
           end
end
end
end
figure(2);imshow(Q);title('水平镜像变换');
%%  缩小和放大
A=imread('wq.jpg'); 
A=uint8(A);
[a,b,c]=size(A);
B=ones(a+100,b+100,c);B=uint8(B);
for n=1:3
for i=1:a+100
for j=1:b+100
     if i<a&&j<b
         B(i,j,n)= A(i,j,n); 
     else
         B(i,j,n)= 255; 
     end
end
end
end
figure(1);imshow(B);title('origin');
Q=ones(a/2,b/2,c);Q=uint8(Q);
for n=1:3
for i=1:a/2
for j=1:b/2          
         Q(i,j,n)= B(2*i,2*j,n);        
end
end
end
figure(2);imshow(Q);title('缩小');

K=ones(2*a,2*b,c);K=uint8(K);
for n=1:3
for i=1:2*a
for j=1:2*b          
         K(i,j,n)= B(ceil(i/2),ceil(j/2),n);    
        if K(i,j,n)==0 
          
           K(i,j,n)=ceil((K(i-1,j,n)+K(i+1,j,n)+K(i,j+1,n)+K(i,j-1,n))/4);
        end
        
end
end
end
figure(3);imshow(K);title('放大');
%%  旋转
A=imread('guku.jpg'); 
A=imresize(A,[320,480]);
rad=pi*30/180;
imshow(A);
[X1,Y1,~]=size(A);
Ymx=max(max(max((Y1-1)*cos(rad)+1,-(X1-1)*sin(rad)+1),(Y1-1)*cos(rad)-(X1-1)*sin(rad)+1),1);
Ymn=min(min(min((Y1-1)*cos(rad)+1,-(X1-1)*sin(rad)+1),(Y1-1)*cos(rad)-(X1-1)*sin(rad)+1),1);
Xmx=max(max(max((Y1-1)*sin(rad)+1,(X1-1)*cos(rad)+1),(Y1-1)*sin(rad)+(X1-1)*cos(rad)+1),1);
Xmn=min(min(min((Y1-1)*sin(rad)+1,(X1-1)*cos(rad)+1),(Y1-1)*sin(rad)+(X1-1)*cos(rad)+1),1);
xtran=0;
ytran=0;
if Xmn<1
xtran=ceil(-(Xmn-1));
end
if Ymn<1
ytran=ceil(-(Ymn-1));
end
MB=zeros(ceil(Xmx-Xmn),ceil(Ymx-Ymn),3);
MB=uint8(MB);
m=MB;
a2=ceil(Xmx-Xmn);
[X2,Y2,~]=size(MB);
for i2=1:X2
for j2=1:Y2
i=ceil((i2-xtran-1)*cos(-rad)+(j2-1-ytran-1)*sin(-rad)+1);
j=ceil(-(i2-1-xtran)*sin(-rad)+(j2-1-ytran)*cos(-rad)+1);
if i>=1 && j>=1 && i<X1 && j<Y1
MB(i2,j2,:)=A(i,j,:);
end
end
end
figure(2);
imshow(MB);
%%  实验三
%% ⒈	实现对数增强或指数增强。
%我采用的指数灰度变换是 g(i,j,n)=2^((g(i,j,n)-30))-1;
%%我采用的对数灰度变换是 g(i,j,n)=41*(log(i,j,n)+1))

g=imread('2.jpg');
g=double(g);
k=g;
[Y,X,Z]=size(g);
for n=1:3
for i=1:Y
for j=1:X
if(g(i,j,n)<200&&g(i,j,n)>30)
g(i,j,n)=2^((g(i,j,n) -30)) -1;
k(i,j,n)=41*(log(k(i,j,n)+1));
end
end
end
end
g=uint8(g);
k=uint8(k);
figure,imshow(g);
title(' 指数灰度变换 ');
figure,imshow(k);
title(' 对数灰度变换 ');
%% ⒉	实现图像直方图均衡化增强。
c=imread('test.png');
c=rgb2gray(c);
[y,x] = size(c);
figure;
imshow(c);
figure;
imhist(c);
N = zeros(1,256);
for i = 1:y
    for j = 1: x
        N(c(i,j) + 1) = N(c(i,j) + 1) + 1;
    end
end
P = zeros(1,256);
for i = 1:256
    P(i) = N(i) / (y * x * 1.0);
end
C = zeros(1,256);
 C(1) = P(1);
for i = 2:256
    C(i) = C(i - 1) + P(i);
end
C= uint8(255 * C + 0.5);
for i = 1:y
    for j = 1: x
        c(i,j) = C(c(i,j));
    end
end
figure;
imshow(c);
figure;
imhist(c);
%%  实验四
%选择测试图像分别添加高斯、椒盐、泊松噪声，实现中值滤波； 
%选择测试图像实现两种常用梯度算子（Sobel算子、Prewitt算子）
%选择测试图像实现理想低通滤波；  
%选择测试图像实现巴特沃斯高通滤波。
%% 选择测试图像分别添加高斯、椒盐、泊松噪声，实现中值滤波； 
C=imread('guku.jpg');
C=imresize(C,[320,480]);  
figure;
x=uint8(C);
x=rgb2gray(x);
j=imnoise(x,'salt & pepper',0.2);
q=imnoise(x,'gaussian',0.01,0.01);
b=imnoise(x,'poisson');
y=mid_filter(q);
z=mid_filter(j);
m=mid_filter(b);
n=ditong(x);
l=batewosi(x);
subplot(3,3,1),imshow(q,[]);title('高斯噪声','FontSize',10);   
subplot(3,3,2),imshow(j,[]);title('椒盐噪声','FontSize',10);   
subplot(3,3,3),imshow(b,[]);title('泊松噪声','FontSize',10);  
subplot(3,3,4),imshow(x,[]);title('原图','FontSize',10);  
subplot(3,3,5),imshow(y,[]);title('高斯后中值','FontSize',10);   
subplot(3,3,6),imshow(z,[]);title('椒盐后中值','FontSize',10);   
subplot(3,3,7),imshow(m,[]);title('泊松后中值','FontSize',10);   
subplot(3,3,8),imshow(n,[]);title('原图进行理想低通','FontSize',10);   
subplot(3,3,9),imshow(l,[]);title('原图巴特沃思','FontSize',10);   
%%  Sobel算子
g=imread('2.jpg');
I = rgb2gray(g);  
subplot(2,2,1);   
imshow(I);    
title('原图');   
 hx=[-1 -2 -1;0 0 0 ;1 2 1];  
 hy=hx';                            
 gradx=filter2(hx,I,'same');  
 gradx=abs(gradx); 
 subplot(2,2,2);  
 imshow(gradx,[]);  
 title('sobel水平梯度');  
 grady=filter2(hy,I,'same');  
 grady=abs(grady); 
 subplot(2,2,3);  
 imshow(grady,[]);  
  title('sobel垂直梯度');  
  grad=gradx+grady;  
  subplot(2,2,4);  
  imshow(grad,[]);  
  title('sobel梯度');  
  [y,x]=size(grad);
  for i=1:y
      for j=1:x
          if grad(i,j)>50
              grad(i,j)=255;
          else    
             grad(i,j)=0;
          end
      end
  end
  figure(2);
   imshow(grad,[]);  
  title('sobel锐化后直接取阈值进行二值化');  
  %%  Roberts算子
g=imread('2.jpg');
I = rgb2gray(g);  
subplot(2,2,1);   
imshow(I);    
title('原图');   
 hx=[0 -1;1 0];  
 hy=hx';                            
 gradx=filter2(hx,I,'same');  
 gradx=abs(gradx); 
 subplot(2,2,2);  
 imshow(gradx,[]);  
 title('roberts 水平梯度');  
 grady=filter2(hy,I,'same');  
 grady=abs(grady); 
 subplot(2,2,3);  
 imshow(grady,[]);  
  title('roberts垂直梯度');  
  grad=gradx+grady;  
  subplot(2,2,4);  
  imshow(grad,[]);  
  title('roberts梯度');  
    [y,x]=size(grad);
  for i=1:y
  for j=1:x
  if grad(i,j)>10
      grad(i,j)=255;
          else    
             grad(i,j)=0;
          end
      end
  end
  figure(2);
   imshow(grad,[]);  
  title('roberts锐化后直接取阈值进行二值化');  
%% Prewitt
g=imread('2.jpg');
I = rgb2gray(g);  
subplot(2,2,1);   
imshow(I);    
title('原图');   
 hx=[-1 -1 -1;0 0 0 ;1 1 1];  
 hy=hx';                            
 gradx=filter2(hx,I,'same');  
 gradx=abs(gradx); 
 subplot(2,2,2);  
 imshow(gradx,[]);  
 title('prewitt水平梯度');  
 grady=filter2(hy,I,'same');  
 grady=abs(grady); 
 subplot(2,2,3);  
 imshow(grady,[]);  
  title('prewitt垂直梯度');  
  grad=gradx+grady;  
  subplot(2,2,4);  
  imshow(grad,[]);  
  title('prewitt梯度'); 
  [y,x]=size(grad);
  for i=1:y
      for j=1:x
          if grad(i,j)>50
              grad(i,j)=255;
          else    
             grad(i,j)=0;

          end
      end
  end
  figure(2);
   imshow(grad,[]);  
  title('ｐｒｅｗｉｔｔ锐化后直接取阈值进行二值化');  
%%  实现三种梯度算子(Roberts、Sobel、Prewitt)的边缘检测。
I = imread('eight.tif');
BW1=edge(I,'roberts');  
figure;  
subplot(2,2,1);
imshow(I);  title('原图');  
subplot(2,2,2);
imshow(BW1);  title('Roberts');  
BW2=edge(I,'sobel');  
subplot(2,2,3);  
imshow(BW2);   title('sobel'); 
BW3=edge(I,'prewitt');  
subplot(2,2,4);  
imshow(BW3);  title('prewitt');  
I = imread('cell.tif');
figure;
imshow(I);
[~, threshold] = edge(I, 'sobel');
fudgeFactor = .5;
BWs = edge(I,'sobel', threshold * fudgeFactor);
% Step 3: 创建用于膨胀的算子
se90 = strel('line', 3, 90);
se0 = strel('line', 3, 0);
% 对图像进行膨胀
BWsdil = imdilate(BWs, [se90 se0]);
%Step 4: Fill Interior Gaps 用imfill填充孔洞
BWdfill = imfill(BWsdil, 'holes');
% Step 5: 去掉边界连接的部分
BWnobord = imclearborder(BWdfill, 4);
%Step 6: 使图像平滑
seD = strel('diamond',1);
BWfinal = imerode(BWnobord,seD);
BWfinal = imerode(BWfinal,seD);
BWoutline = bwperim(BWfinal);
[y,x,z]=size(I);
a=zeros(y,x,z);
%Segout = I; 
Segout=a;
Segout(BWoutline) = 255; 
figure, imshow(Segout), title('sobel边缘检测');
%% 实现对一张二值图像的边界跟踪。 
img=imread('eight.tif');
img=img>128;
[m n]=size(img);
imgn=zeros(m,n);      
ed=[-1 -1;0 -1;1 -1;1 0;1 1;0 1;-1 1;-1 0];
for i=2:m-1
    for j=2:n-1
        if img(i,j)==1      
            for k=1:8
                ii=i+ed(k,1);
                jj=j+ed(k,2);
                if img(ii,jj)==0   
                    imgn(ii,jj)=1;
                end
            end
            
        end
    end
end
figure;
imshow(imgn,[]);
%% 实现分水岭算法。
rgb = imread('pears.png');
I = rgb2gray(rgb);
hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(I), hy, 'replicate');
Ix = imfilter(double(I), hx, 'replicate');  
gradmag = sqrt(Ix.^2 + Iy.^2);
%以上先用sobel算子提取轮廓
se = strel('disk', 20);
Io = imopen(I, se);
Ie = imerode(I, se);
Iobr = imreconstruct(Ie, I);
Iobrd = imdilate(Iobr, se);
Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
fgm = imregionalmax(Iobrcbr);
I2 = I;
I2(fgm) = 255;
se2 = strel(ones(5,5));
fgm2 = imclose(fgm, se2);
fgm3 = imerode(fgm2, se2);
fgm4 = bwareaopen(fgm3, 20);
I3 = I;
I3(fgm4) = 255;
bw = im2bw(Iobrcbr, graythresh(Iobrcbr));
D = bwdist(bw);
DL = watershed(D);
bgm = DL == 0;
gradmag2 = imimposemin(gradmag, bgm | fgm4);
L = watershed(gradmag2);
I4 = I;
I4(imdilate(L == 0, ones(3, 3)) | bgm | fgm4) = 255;
Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');
figure
imshow(I)
hold on
himage = imshow(Lrgb);
himage.AlphaData = 0.3;
title('将分水岭分割后图像透明叠加在原图像上')
%%  实验六
%实现背景差值法对运动目标的检测；
%实现背景的更新。
trafficVid = VideoReader('jiebai.avi');
I = read(trafficVid, 70);
back=read(trafficVid, 1);
c=I-back;
figure;imshow(c);

%% Detecting Cars in a Video of Traffic 流量统计
%训练
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 10);
total=0;
oldnum=0;
videoReader = vision.VideoFileReader('1.mp4');
for i = 1:100
    frame = step(videoReader); % read the next video frame
    foreground = step(foregroundDetector, frame);
end
se = strel('square', 3);
filteredForeground = imopen(foreground, se);
%figure; imshow(filteredForeground); title('Clean Foreground');
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 2000);
bbox = step(blobAnalysis, filteredForeground);
    for i=1:size(bbox,1)
      if bbox(i,1)<60 ||bbox(i,2)<170 
        bbox(i,:)=[];
      end   
      if size(bbox,1)==1 
          break;
      end   
    end   
result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');
%四车道 合理范围：x :205-570 y: 50-300
numCars = size(bbox, 1);
result = insertText(result, [10 10], numCars, 'BoxOpacity', 1, ...
    'FontSize', 14);
%figure; imshow(result); title('Detected Cars');
% 对整个视频进行处理
videoReader = vision.VideoFileReader('jiebai.avi');
videoPlayer = vision.VideoPlayer('Name', 'Detected Cars');
videoPlayer.Position(3:4) = [650,400];  % window size: [width, height]
se = strel('square', 3); % morphological filter for noise removal
while ~isDone(videoReader)
    frame = step(videoReader); 
    foreground = step(foregroundDetector,frame);
    filteredForeground = imopen(foreground, se);
    bbox = step(blobAnalysis, filteredForeground);
    for i=1:size(bbox,1)
      if bbox(i,1)<60 ||bbox(i,2)<200 
        bbox(i,:)=[];
      end   
      if size(bbox,1)<=i
          break;
      end   
    end   
    result = insertShape(frame, 'Rectangle', bbox, 'Color', 'Yellow');
    numCars = size(bbox, 1);
    if(numCars>oldnum)
        total=total+numCars-oldnum;
    end
    result = insertText(result, [10 10], 'now:', 'BoxOpacity', 1, ...
        'FontSize', 14);
     result = insertText(result, [50 10],numCars, 'BoxOpacity', 1, ...
        'FontSize', 14);
     result = insertText(result, [10 50], 'total:', 'BoxOpacity', 1, ...
        'FontSize', 14);
     result = insertText(result, [50 50],total, 'BoxOpacity', 1, ...
        'FontSize', 14);
    oldnum=numCars;
    step(videoPlayer, result);  % display the results
end
release(videoReader); % close the video file
%% 安防监控运动目标检测算法设计及实现
%1）完成视频源的帧化。
%2）对其中一帧通过背景差分法进行目标检测。
% a = VideoReader('2.mp4'); %1296 total
% nframes = a.NumberOfFrames;
% for k = 1 : nframes
% 
%         singleFrame = read(a, k);
%         imshow(singleFrame);
% end
% back=read(a, 1);
% now=(read(a,200));
% c=now-back;
% imshow(c);
videoReader = vision.VideoFileReader('2.mp4');
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 10);
for i = 1:1000
    frame = step(videoReader); % read the next video frame
    foreground = step(foregroundDetector, frame);
end
se = strel('square', 3);
filteredForeground = imopen(foreground, se);
figure; imshow(filteredForeground); title('Clean Foreground');
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 200);
bbox = step(blobAnalysis, filteredForeground);
result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');
numCars = size(bbox, 1);
figure; imshow(result); title('运动物体追踪');
videoPlayer = vision.VideoPlayer('Name', 'Detected Cars');
i=0;
while ~isDone(videoReader) 
    frame = step(videoReader); 
    foreground = step(foregroundDetector, frame);
    filteredForeground = imopen(foreground, se);  
    bbox = step(blobAnalysis, filteredForeground);
      if size(bbox,1)>1
            bbox=bbox(2,:);
      end   
    result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');
    numCars = size(bbox, 1);
    step(videoPlayer, result); 
  i=i+1;
    if i>200
        break;
    end
end
release(videoReader); 
%% 目标跟踪
videoReader = vision.VideoFileReader('a.mp4');
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 10);
se = strel('square', 3);
videoPlayer = vision.VideoPlayer('Name', 'Detected Cars');
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 50);
while ~isDone(videoReader) 
    step(videoReader);
    step(videoReader);
    frame = step(videoReader);
    foreground = step(foregroundDetector, frame);
    filteredForeground = imopen(foreground, se);  
    bbox = step(blobAnalysis, filteredForeground);
      if size(bbox,1)>1
            bbox=bbox(2,:);
      end   
    result = insertShape(frame, 'Rectangle', bbox, 'Color', 'red');
    numCars = size(bbox, 1);
    step(videoPlayer, result); 
   
end
release(videoReader); 

%% 在给定时长内（自定）对每车道的车辆进行计数。





%% 
%视频帧化
target = VideoReader('aa.mp4');
nframes = target.NumberOfFrames;
for k=114: nframes
frame=read(target,k);
frame=rgb2gray(frame);
imwrite(frame,strcat('I:\MatImg\explore\大实验报告\meanShift-master\zjut\',int2str(k),'.jpg'));

end
%% 图片转视频
framesPath = 'I:\MatImg\explore\zjut\';%图像序列所在路径，同时要保证图像大小相同  
videoName = 'I:\MatImg\explore\zjut\Bolt';%表示将要创建的视频文件的名字  
fps = 25; %帧率  
if(exist('videoName','file'))  
    delete videoName.avi  
end  
%生成视频的参数设定  
aviobj=VideoWriter(videoName);  %创建一个avi视频文件对象，开始时其为空  
aviobj.FrameRate=fps;  
open(aviobj);%Open file for writing video data  
i=1;
%读入图片  
while i<205
    fileName=sprintf('%4d',i);    %根据文件名而定 我这里文件名是0001.jpg 0002.jpg ....  
    %fileName=sprintf('%04d',i);    %根据文件名而定 我这里文件名是0001.jpg 0002.jpg ....  
    frames=imread([framesPath,fileName,'.jpg']);  
    writeVideo(aviobj,frames);  
    i=i+4;
end  
close(aviobj);% 关闭创建视频 
%%
target = VideoReader('I:\MatImg\explore\aa.mp4');%x44-64 y110-150
nframes = target.NumberOfFrames;
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 10);
se = strel('square', 3);
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 10);
for k=1: nframes
frame=read(target,k);
    foreground = step(foregroundDetector, frame);
    filteredForeground = imopen(foreground, se);  
    bbox = step(blobAnalysis, filteredForeground);
      if size(bbox,1)>1
            bbox=bbox(2,:);
      end   
    result = insertShape(frame, 'Rectangle', bbox, 'Color', 'red');
    imshow(result);
end
%%

