    const video = document.getElementById('webcam');
    const liveView = document.getElementById('liveView');
    const demosSection = document.getElementById('demos');
    const enableWebcamButton = document.getElementById('webcamButton');
    const img = document.querySelector("img");
    const imgView = document.getElementById("imgView")
    var webcamCanvas = document.createElement('canvas');
    imgView.append(webcamCanvas)
    let model;
    var children = []
    let VidframeCount = 0;
    let CVframeCount = 0;
    var  lastCalledTime = performance.now()
    var fps = 0
    var predTime = 0

    // Check if webcam access is supported.
    function getUserMediaSupported() {
        return !!(navigator.mediaDevices &&
        navigator.mediaDevices.getUserMedia);
    }
    
    // Enable the live webcam view and start classification.
    function enableCam(event) {
        // Only continue if the COCO-SSD has finished loading.
        if (!model) {
        return;
        }
        
        // Hide the button once clicked.
        event.target.classList.add('removed');  
        
        // getUsermedia parameters to force video but not audio.
        const constraints = {
        video: true
        };
    
        // Activate the webcam stream.
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        video.srcObject = stream;
        video.addEventListener('loadeddata', predictWebcam);
        });
    }

    function predictWebcam() {
        console.log(tf.getBackend());

        webcamCanvas.width = video.videoWidth;
        webcamCanvas.height = video.videoHeight;
        const webcamCanvasCtx = webcamCanvas.getContext('2d');
        webcamCanvasCtx.drawImage(video, 0, 0,video.videoWidth,video.videoHeight);
        // const imageData = webcamCanvasCtx.getImageData(0, 0, video.videoWidth, video.videoHeight);

        tf.tidy(() => {
            let input = tf.image.resizeBilinear(tf.browser.fromPixels(video), [416, 416])
            // console.log(input.shape)
            input = tf.sub(tf.div(tf.expandDims(input), 127.5), 1);
            input = tf.cast(input, 'float32');
            console.log(input)
    
            // const inputTensor = preprocessInput(video);
    
            // const imageData = webcamCanvasCtx.getImageData(0, 0, 416, 416);
            // input = tf.browser.fromPixels(imageData).expandDims()
            var start = performance.now()
            const result = model.predict(input);
            var end = performance.now()
            predTime = end - start
            const outputData = result.arraySync(); // Convert to synchronous operation
            var detectionList = [];
    
            // outputData is now a 3-dimensional array (nested arrays) with shape (1, 6, 3549)
            for (let i = 0; i < outputData[0][0].length; i++) {
                var Baskt = outputData[0][4][i];
                var Rim = outputData[0][5][i];
                var score = Baskt;
                var label = 0;
                if (Rim > Baskt) {
                    score = Rim;
                    label = 1;
                }
                if (score > 0.2) {
                    var xMin = outputData[0][0][i];
                    var yMin = outputData[0][1][i];
                    var Width = outputData[0][2][i];
                    var Height = outputData[0][3][i];
                    var detection = [
                        score,
                        [xMin - Width / 2, yMin - Height / 2, xMin + Width / 2, yMin + Height / 2, label],
                    ];
                    detectionList.push(detection);
                }
            }
            detectionList.sort((a, b) => a[0] - b[0]);
            detectionList = performeNMS(detectionList)
            drawRect(webcamCanvasCtx,detectionList)
        })

        // // Call this function again to keep predicting when the browser is ready.
        window.requestAnimationFrame(predictWebcam);
    }

    function drawRect(ctx,detectionList) {
        VidframeCount++
        const elapsedTime = performance.now() - lastCalledTime
        if (elapsedTime >= 1000) {
            fps = VidframeCount
            VidframeCount = 0
            lastCalledTime = performance.now()
        }
        for (let n = 0; n < detectionList.length; n++) {
            const xmin = detectionList[n][1][0] * video.videoWidth;
            const ymin = detectionList[n][1][1] * video.videoHeight;
            const xmax = detectionList[n][1][2] * video.videoWidth;
            const ymax = detectionList[n][1][3] * video.videoHeight;
            const score = detectionList[n][0];
            

            var text = Math.round(parseFloat(score) * 100) + '% confidence.'
            const color = 'green'
            ctx.strokeStyle = color
            ctx.lineWidth = 5; // Set the line width
            ctx.font = 'bold 18px Aerial'
            ctx.fillStyle = color

            ctx.beginPath()
            ctx.fillText(text,xmin,ymin)
            ctx.rect(xmin,ymin,xmax-xmin,ymax-ymin)
            ctx.stroke()
        }
        const color = 'red';
        ctx.fillStyle = color;
        ctx.font = 'bold 18px Arial';
        ctx.fillText(`PredTime: ${Math.round(predTime)} ms`,0,35)
        ctx.fillText(`FPS: ${fps}`, 0, 18);
    }

    function detect() {
        let input = tf.image.resizeBilinear(tf.browser.fromPixels(img), [416, 416]);
        input = tf.cast(tf.expandDims(input), 'float32');
        const result = model.predict(input);
        console.log(result) 
    }

    function preprocessInput(imageData) {
        const imageTensor = tf.browser.fromPixels(imageData); // Convert the image data to a TensorFlow.js tensor.
        
        // Resize the image to the required input size of your model (e.g., 416x416).
        const resizedImage = tf.image.resizeBilinear(imageTensor, [416, 416]);
      
        // Normalize the pixel values to be within the range [0, 1].
        // const normalizedImage = resizedImage.div(255.0);
      
        // Add an extra dimension to the tensor to match the model's input shape (batch size of 1).
        const inputTensor = resizedImage.expandDims(0);
      
        return inputTensor;
      }
    
      function performeNMS(detectionList) {
        var detectsize = detectionList.length;
    
        for (let i = detectsize - 1; i >= 0; i--) {
            if (i >= detectionList.length) {
                break
            }
            var detection1 = detectionList[i];
    
            for (let j = i - 1; j >= 0; j--) {
                var detection2 = detectionList[j];
    
                if (detection1[1][4] === detection2[1][4]) {
                    var iou = calculateIoU(detection1[1], detection2[1]);
                    if (iou > 0.4) {
                        detectionList.splice(j, 1);
                    }
                }
            }
        }
        return detectionList;
    }
    
    function calculateIoU(box1,box2) {
        const [x1, y1, x1x, y1x, l1] = box1;
        const [x2, y2, x2x, y2x, l2] = box2;

        const w1 = (x1x-x1)
        const h1 = (y1x-y1)
        const w2 = (x2x-x2)
        const h2 = (y2x-y2)

        // Calculate coordinates of intersection rectangle
        const xLeft = Math.max(x1, x2)
        const yTop = Math.max(y1, y2)
        const xRight = Math.min(x1x, x2x)
        const yBottom = Math.min(y1x, y2x)

        // Calculate intersection area
        const intersectionArea = Math.max(0, xRight - xLeft) * Math.max(0, yBottom - yTop)

        // Calculate union area
        const box1Area = w1 * h1
        const box2Area = w2 * h2
        const unionArea = box1Area + box2Area - intersectionArea

        // Calculate IoU
        const iou = intersectionArea / unionArea

        return iou
    }

    async function loadmodel() {
        const modelPath = 'TFLite/best25_integer_quant.tflite'
        if (!model) {
            model = await tflite.loadTFLiteModel(modelPath)
            // Warmup the model before using real data.
            const warmupResult = model.predict(tf.zeros([1,416,416,3]));
            warmupResult.dataSync();
            warmupResult.dispose();
            console.log("WarmUpDone")
            demosSection.classList.remove('invisible');
        }
    }

    function init() {
        var children = []
        // If webcam supported, add event listener to button for when user
        if (getUserMediaSupported()) {
            loadmodel()
            // enableWebcamButton.addEventListener('click', detect);
            enableWebcamButton.addEventListener('click', enableCam);
        } else {
            console.warn('getUserMedia() is not supported by your browser');
        }
    }

    init()
    