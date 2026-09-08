let totalDetected = 0;
let totalWithMask = 0;
let totalWithoutMask = 0;


const startButton =
    document.getElementById("startCamera");

const stopButton =
    document.getElementById("stopCamera");

const videoFeed =
    document.getElementById("videoFeed");

const cameraPlaceholder =
    document.getElementById("cameraPlaceholder");

const cameraStatus =
    document.getElementById("cameraStatus");

const uploadForm =
    document.getElementById("uploadForm");

const imageInput =
    document.getElementById("imageInput");

const selectedFile =
    document.getElementById("selectedFile");

const loading =
    document.getElementById("loading");

const resultSection =
    document.getElementById("resultSection");

const resultImage =
    document.getElementById("resultImage");

const resultDetails =
    document.getElementById("resultDetails");

const totalCount =
    document.getElementById("totalCount");

const maskCount =
    document.getElementById("maskCount");

const noMaskCount =
    document.getElementById("noMaskCount");


/* =======================================================
   START CAMERA
======================================================= */

startButton.addEventListener(
    "click",
    async () => {

        try {

            const response = await fetch(
                "/start-camera",
                {
                    method: "POST"
                }
            );

            const data =
                await response.json();

            if (!data.success) {

                alert(data.message);

                return;
            }

            videoFeed.src =
                "/video-feed?" + Date.now();

            videoFeed.style.display =
                "block";

            cameraPlaceholder.style.display =
                "none";

            cameraStatus.textContent =
                "LIVE";

            cameraStatus.className =
                "online";

            startButton.disabled =
                true;

            stopButton.disabled =
                false;

        } catch (error) {

            console.error(error);

            alert(
                "Unable to start camera."
            );
        }
    }
);


/* =======================================================
   STOP CAMERA
======================================================= */

stopButton.addEventListener(
    "click",
    async () => {

        await fetch(
            "/stop-camera",
            {
                method: "POST"
            }
        );

        videoFeed.src = "";

        videoFeed.style.display =
            "none";

        cameraPlaceholder.style.display =
            "flex";

        cameraPlaceholder.style.flexDirection =
            "column";

        cameraStatus.textContent =
            "OFFLINE";

        cameraStatus.className =
            "offline";

        startButton.disabled =
            false;

        stopButton.disabled =
            true;
    }
);


/* =======================================================
   SELECT IMAGE
======================================================= */

imageInput.addEventListener(
    "change",
    () => {

        if (imageInput.files.length > 0) {

            selectedFile.textContent =
                "Selected: " +
                imageInput.files[0].name;
        }
    }
);


/* =======================================================
   IMAGE DETECTION
======================================================= */

uploadForm.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();

        if (
            imageInput.files.length === 0
        ) {

            alert(
                "Please select an image."
            );

            return;
        }

        const formData =
            new FormData();

        formData.append(
            "image",
            imageInput.files[0]
        );

        loading.classList.remove(
            "hidden"
        );

        resultSection.classList.add(
            "hidden"
        );

        try {

            const response = await fetch(
                "/predict-image",
                {
                    method: "POST",
                    body: formData
                }
            );

            const data =
                await response.json();

            loading.classList.add(
                "hidden"
            );

            if (!data.success) {

                alert(data.message);

                return;
            }

            resultImage.src =
                data.image_url +
                "?" +
                Date.now();

            // Add current detection results to cumulative counters

            totalDetected += data.total;
            totalWithMask += data.mask;
            totalWithoutMask += data.no_mask;


            // Update dashboard

            totalCount.textContent =
                totalDetected;

            maskCount.textContent =
                totalWithMask;

            noMaskCount.textContent =
                totalWithoutMask;

            resultDetails.innerHTML = "";

            if (
                data.results.length === 0
            ) {

                resultDetails.innerHTML =
                    `
                    <div class="detection-result">
                        No face detected.
                    </div>
                    `;

            } else {

                data.results.forEach(
                    (result, index) => {

                        resultDetails.innerHTML +=
                            `
                            <div class="detection-result">

                                <strong>
                                    Face ${index + 1}
                                </strong>

                                <br>

                                ${result.label}

                                —

                                ${result.confidence}%

                            </div>
                            `;
                    }
                );
            }

            resultSection.classList.remove(
                "hidden"
            );

        } catch (error) {

            loading.classList.add(
                "hidden"
            );

            console.error(error);

            alert(
                "Detection failed."
            );
        }
    }
);