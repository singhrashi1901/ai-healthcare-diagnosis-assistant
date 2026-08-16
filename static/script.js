/* =====================================================
   HEALTHAI - MAIN JAVASCRIPT
===================================================== */


/* =====================================================
   DOM READY
===================================================== */

document.addEventListener("DOMContentLoaded", () => {

    /* =========================
       LOAD SAVED THEME
    ========================= */

    const savedTheme = localStorage.getItem("theme");
    const themeBtn = document.getElementById("themeBtn");

    if (savedTheme === "dark") {

        document.body.classList.add("dark");

        if (themeBtn) {
            themeBtn.innerHTML = "☀️";
        }

    } else {

        if (themeBtn) {
            themeBtn.innerHTML = "🌙";
        }
    }


    /* =========================
       FILE INPUT
    ========================= */

    const fileInput =
        document.getElementById("reportFile");

    if (fileInput) {

        fileInput.addEventListener("change", () => {

            if (fileInput.files.length > 0) {

                const file =
                    fileInput.files[0];

                showUploadMessage(
                    "✓ " + file.name +
                    " selected successfully!",
                    "success"
                );

            }

        });

    }


    /* =========================
       DRAG & DROP
    ========================= */

    const uploadCard =
        document.querySelector(".upload-card");

    if (uploadCard) {

        uploadCard.addEventListener(
            "dragover",
            (event) => {

                event.preventDefault();

                uploadCard.classList.add(
                    "drag-active"
                );

            }
        );


        uploadCard.addEventListener(
            "dragleave",
            () => {

                uploadCard.classList.remove(
                    "drag-active"
                );

            }
        );


        uploadCard.addEventListener(
            "drop",
            (event) => {

                event.preventDefault();

                uploadCard.classList.remove(
                    "drag-active"
                );

                const files =
                    event.dataTransfer.files;

                if (files.length > 0) {

                    const file = files[0];

                    const allowedTypes = [
                        "application/pdf",
                        "image/jpeg",
                        "image/png"
                    ];

                    if (
                        !allowedTypes.includes(
                            file.type
                        )
                    ) {

                        showUploadMessage(
                            "⚠️ Please upload PDF, JPG, JPEG or PNG.",
                            "error"
                        );

                        return;
                    }


                    fileInput.files = files;

                    showUploadMessage(
                        "✓ " + file.name +
                        " selected successfully!",
                        "success"
                    );

                }

            }
        );

    }

});


/* =====================================================
   DAY / NIGHT MODE
===================================================== */

function toggleTheme() {

    document.body.classList.toggle("dark");

    const themeBtn =
        document.getElementById("themeBtn");


    if (
        document.body.classList.contains(
            "dark"
        )
    ) {

        if (themeBtn) {
            themeBtn.innerHTML = "☀️";
        }

        localStorage.setItem(
            "theme",
            "dark"
        );

    } else {

        if (themeBtn) {
            themeBtn.innerHTML = "🌙";
        }

        localStorage.setItem(
            "theme",
            "light"
        );

    }

}


/* =====================================================
   EXAMPLE SYMPTOMS
===================================================== */

function addExample(text) {

    const textarea =
        document.getElementById(
            "symptoms"
        );

    if (!textarea) return;


    if (
        textarea.value.trim() === ""
    ) {

        textarea.value = text;

    } else {

        textarea.value +=
            ", " + text;

    }


    textarea.focus();

}


/* =====================================================
   AI SYMPTOM ANALYSIS
===================================================== */

async function analyzeSymptoms() {

    const symptomsInput =
        document.getElementById(
            "symptoms"
        );

    const loading =
        document.getElementById(
            "loading"
        );

    const result =
        document.getElementById(
            "result"
        );

    const button =
        document.getElementById(
            "analyzeBtn"
        );


    if (!symptomsInput) return;


    const text =
        symptomsInput.value.trim();


    /* =========================
       EMPTY INPUT
    ========================= */

    if (!text) {

        showNotification(
            "Please describe your symptoms first.",
            "warning"
        );

        symptomsInput.focus();

        return;
    }


    /* =========================
       MINIMUM TEXT
    ========================= */

    if (text.length < 3) {

        showNotification(
            "Please enter a little more information about your symptoms.",
            "warning"
        );

        return;
    }


    /* =========================
       HIDE OLD RESULT
    ========================= */

    if (result) {

        result.classList.add(
            "hidden"
        );

    }


    /* =========================
       SHOW LOADING
    ========================= */

    if (loading) {

        loading.classList.remove(
            "hidden"
        );

    }


    if (button) {

        button.disabled = true;

        button.innerHTML = `
            <span class="button-spinner"></span>
            Analyzing...
        `;

    }


    try {

        const response =
            await fetch(
                "/predict",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        symptoms: text
                    })

                }
            );


        if (!response.ok) {

            throw new Error(
                "Server returned status " +
                response.status
            );

        }


        const data =
            await response.json();


        /* =========================
           BACKEND ERROR
        ========================= */

        if (!data.success) {

            throw new Error(
                data.message ||
                "Unable to analyze symptoms."
            );

        }


        /* =========================
           DISEASE
        ========================= */

        const diseaseElement =
            document.getElementById(
                "disease"
            );


        if (diseaseElement) {

            diseaseElement.innerText =
                formatDiseaseName(
                    data.disease
                );

        }


        /* =========================
           DETECTED SYMPTOMS
        ========================= */

        const symptomContainer =
            document.getElementById(
                "detectedSymptoms"
            );


        if (symptomContainer) {

            symptomContainer.innerHTML =
                "";


            if (
                data.symptoms &&
                Array.isArray(data.symptoms) &&
                data.symptoms.length > 0
            ) {

                data.symptoms.forEach(
                    symptom => {

                        const tag =
                            document.createElement(
                                "span"
                            );

                        tag.className =
                            "symptom-tag";

                        tag.innerHTML =
                            "✓ " +
                            formatDiseaseName(
                                symptom
                            );

                        symptomContainer
                            .appendChild(tag);

                    }
                );

            } else {

                symptomContainer.innerHTML =
                    `
                    <span class="symptom-tag">
                        No specific symptoms detected
                    </span>
                    `;

            }

        }


        /* =========================
           SHOW RESULT
        ========================= */

        if (result) {

            result.classList.remove(
                "hidden"
            );


            setTimeout(() => {

                result.scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });

            }, 100);

        }


        showNotification(
            "✓ AI analysis completed!",
            "success"
        );


    } catch (error) {

        console.error(
            "HealthAI Error:",
            error
        );


        showNotification(
            error.message ||
            "Unable to connect to HealthAI.",
            "error"
        );


    } finally {

        /* =========================
           HIDE LOADING
        ========================= */

        if (loading) {

            loading.classList.add(
                "hidden"
            );

        }


        /* =========================
           RESET BUTTON
        ========================= */

        if (button) {

            button.disabled = false;

            button.innerHTML = `
                🧠 Analyze Symptoms
                <span>→</span>
            `;

        }

    }

}


/* =====================================================
   FORMAT TEXT
===================================================== */

function formatDiseaseName(text) {

    if (!text) {
        return "Unknown";
    }


    return text
        .toString()
        .replace(/\b\w/g, letter =>
            letter.toUpperCase()
        );

}


/* =====================================================
   MEDICAL REPORT ANALYSIS
===================================================== */

async function uploadReport() {

    const fileInput =
        document.getElementById(
            "reportFile"
        );


    const messageElement =
        document.getElementById(
            "uploadMessage"
        );


    const result =
        document.getElementById(
            "reportResult"
        );


    const resultText =
        document.getElementById(
            "reportText"
        );


    const loading =
        document.getElementById(
            "reportLoading"
        );


    const analyzeButton =
        document.getElementById(
            "analyzeReportBtn"
        );


    /* =========================
       CHECK FILE
    ========================= */

    if (
        !fileInput ||
        !fileInput.files.length
    ) {

        showUploadMessage(
            "⚠️ Please select a medical report first.",
            "error"
        );

        return;

    }


    const file =
        fileInput.files[0];


    /* =========================
       VALID EXTENSIONS
    ========================= */

    const allowedExtensions = [
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png"
    ];


    const fileName =
        file.name.toLowerCase();


    const validFile =
        allowedExtensions.some(
            extension =>
                fileName.endsWith(
                    extension
                )
        );


    if (!validFile) {

        showUploadMessage(
            "⚠️ Please upload PDF, JPG, JPEG or PNG.",
            "error"
        );

        return;

    }


    /* =========================
       FILE SIZE
    ========================= */

    const maxSize =
        10 * 1024 * 1024;


    if (file.size > maxSize) {

        showUploadMessage(
            "⚠️ File size must be less than 10 MB.",
            "error"
        );

        return;

    }


    /* =========================
       SELECTED MESSAGE
    ========================= */

    showUploadMessage(
        "✓ " + file.name +
        " selected successfully!",
        "success"
    );


    /* =========================
       LOADING UI
    ========================= */

    if (analyzeButton) {

        analyzeButton.disabled =
            true;

        analyzeButton.innerHTML = `
            <span class="button-spinner"></span>
            Analyzing Report...
        `;

    }


    if (loading) {

        loading.classList.remove(
            "hidden"
        );

    }


    if (result) {

        result.classList.add(
            "hidden"
        );

    }


    /* =========================
       FORM DATA
    ========================= */

    const formData =
        new FormData();


    formData.append(
        "report",
        file
    );


    try {

        /* =========================
           SEND TO FLASK
        ========================= */

        const response =
            await fetch(
                "/analyze-report",
                {

                    method: "POST",

                    body: formData

                }
            );


        const data =
            await response.json();


        /* =========================
           SERVER ERROR
        ========================= */

        if (
            !response.ok ||
            !data.success
        ) {

            throw new Error(
                data.message ||
                "Medical report analysis failed."
            );

        }


        /* =================================================
           BUILD REPORT RESULT
        ================================================= */

        let html = "";


        /* =========================
           POSSIBLE CONDITION
        ========================= */

        if (data.disease) {

            html += `
                <div class="report-condition">

                    <small>
                        POSSIBLE CONDITION
                    </small>

                    <h2>
                        🩺
                        ${formatDiseaseName(
                            data.disease
                        )}
                    </h2>

                </div>
            `;

        } else {

            html += `
                <div class="report-condition">

                    <small>
                        AI REPORT ANALYSIS
                    </small>

                    <h3>
                        ℹ️ No condition predicted
                    </h3>

                    <p>
                        No recognizable symptoms
                        from the trained dataset
                        were detected.
                    </p>

                </div>
            `;

        }


        /* =========================
           DETECTED SYMPTOMS
        ========================= */

        html += `
            <div class="report-detected">

                <h3>
                    🔍 Detected Symptoms
                </h3>

                <div class="report-tags">
        `;


        if (
            data.symptoms &&
            Array.isArray(data.symptoms) &&
            data.symptoms.length > 0
        ) {

            data.symptoms.forEach(
                symptom => {

                    html += `
                        <span class="symptom-tag">
                            ✓
                            ${formatDiseaseName(
                                symptom
                            )}
                        </span>
                    `;

                }
            );

        } else {

            html += `
                <span class="symptom-tag">
                    No recognizable symptoms detected
                </span>
            `;

        }


        html += `
                </div>

            </div>
        `;


        /* =========================
           EXTRACTED REPORT TEXT
        ========================= */

        html += `
            <div class="report-text-preview">

                <h3>
                    📄 Extracted Report Text
                </h3>

                <div class="extracted-text">

                    ${escapeHtml(
                        data.report_text ||
                        "No readable text was extracted."
                    )}

                </div>

            </div>
        `;


        /* =========================
           INSERT RESULT
        ========================= */

        if (resultText) {

            resultText.innerHTML =
                html;

        }


        if (result) {

            result.classList.remove(
                "hidden"
            );


            setTimeout(() => {

                result.scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });

            }, 100);

        }


        showUploadMessage(
            "✓ Report analyzed successfully!",
            "success"
        );


        showNotification(
            "✓ Medical report analysis completed!",
            "success"
        );


    } catch (error) {

        console.error(
            "Medical Report Error:",
            error
        );


        showNotification(
            error.message ||
            "Unable to analyze the medical report.",
            "error"
        );

    } finally {

        /* =========================
           HIDE LOADING
        ========================= */

        if (loading) {

            loading.classList.add(
                "hidden"
            );

        }


        /* =========================
           RESET BUTTON
        ========================= */

        if (analyzeButton) {

            analyzeButton.disabled =
                false;

            analyzeButton.innerHTML =
                `
                🧠 Analyze Medical Report
                `;

        }

    }

}


/* =====================================================
   ALIAS FOR ANALYZE BUTTON
===================================================== */

async function analyzeReport() {

    await uploadReport();

}


/* =====================================================
   ESCAPE HTML
===================================================== */

function escapeHtml(text) {

    const div =
        document.createElement(
            "div"
        );


    div.textContent =
        text || "";


    return div.innerHTML;

}


/* =====================================================
   UPLOAD MESSAGE
===================================================== */

function showUploadMessage(
    message,
    type
) {

    const messageElement =
        document.getElementById(
            "uploadMessage"
        );


    if (!messageElement) {
        return;
    }


    messageElement.innerText =
        message;


    messageElement.className =
        "upload-message " + type;

}


/* =====================================================
   NOTIFICATION
===================================================== */

function showNotification(
    message,
    type = "info"
) {

    /* Remove old notification */

    const oldNotification =
        document.querySelector(
            ".healthai-notification"
        );


    if (oldNotification) {

        oldNotification.remove();

    }


    /* Create notification */

    const notification =
        document.createElement(
            "div"
        );


    notification.className =
        `healthai-notification ${type}`;


    let icon = "ℹ️";


    if (type === "error") {
        icon = "❌";
    }


    if (type === "warning") {
        icon = "⚠️";
    }


    if (type === "success") {
        icon = "✓";
    }


    notification.innerHTML = `
        <span class="notification-icon">
            ${icon}
        </span>

        <span>
            ${message}
        </span>

        <button
            onclick="this.parentElement.remove()">
            ×
        </button>
    `;


    document.body.appendChild(
        notification
    );


    /* Auto remove */

    setTimeout(() => {

        if (notification) {

            notification.remove();

        }

    }, 5000);

}


/* =====================================================
   NAVBAR ACTIVE LINK
===================================================== */

window.addEventListener(
    "scroll",
    () => {

        const sections =
            document.querySelectorAll(
                "section[id]"
            );


        const navLinks =
            document.querySelectorAll(
                ".nav-links a"
            );


        let currentSection = "";


        sections.forEach(
            section => {

                const sectionTop =
                    section.offsetTop -
                    150;


                const sectionHeight =
                    section.offsetHeight;


                if (
                    window.scrollY >=
                    sectionTop &&

                    window.scrollY <
                    sectionTop +
                    sectionHeight
                ) {

                    currentSection =
                        section.getAttribute(
                            "id"
                        );

                }

            }
        );


        navLinks.forEach(
            link => {

                link.classList.remove(
                    "active"
                );


                if (
                    link.getAttribute(
                        "href"
                    ) ===
                    "#" +
                    currentSection
                ) {

                    link.classList.add(
                        "active"
                    );

                }

            }
        );

    }
);


/* =====================================================
   SMOOTH SCROLL
===================================================== */

document.addEventListener(
    "click",
    (event) => {

        const link =
            event.target.closest(
                'a[href^="#"]'
            );


        if (!link) return;


        const targetId =
            link.getAttribute(
                "href"
            );


        if (targetId === "#") {
            return;
        }


        const target =
            document.querySelector(
                targetId
            );


        if (target) {

            event.preventDefault();


            target.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        }

    }
);


/* =====================================================
   CTRL + ENTER
===================================================== */

document.addEventListener(
    "keydown",
    (event) => {

        const textarea =
            document.getElementById(
                "symptoms"
            );


        if (
            document.activeElement ===
            textarea &&

            event.ctrlKey &&

            event.key === "Enter"
        ) {

            analyzeSymptoms();

        }

    }
);
// ============================================================
// MEDICAL REPORT ANALYSIS
// ============================================================

async function analyzeReport() {

    const fileInput = document.getElementById("reportFile");
    const message = document.getElementById("uploadMessage");

    const result = document.getElementById("reportResult");
    const resultText = document.getElementById("reportText");

    const loading = document.getElementById("reportLoading");
    const button = document.getElementById("analyzeReportBtn");

    if (!fileInput.files.length) {

        message.innerText = "⚠️ Please select a medical report first.";

        return;
    }

    const file = fileInput.files[0];

    const formData = new FormData();

    formData.append("report", file);

    message.innerText = "📄 " + file.name + " selected.";

    result.classList.add("hidden");

    loading.classList.remove("hidden");

    button.disabled = true;

    button.innerText = "🧠 Analyzing...";

    try {

        const response = await fetch(
            "/analyze-report",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!data.success) {

            alert(data.message || "Report analysis failed.");

            return;
        }


        // ====================================================
        // DISPLAY RESULT
        // ====================================================

        let output = "";

        output += "<strong>AI Analysis:</strong><br><br>";

        output += data.message + "<br><br>";


        if (data.disease) {

            output +=
                "<strong>Possible Condition:</strong> " +
                data.disease +
                "<br><br>";

        }


        if (data.symptoms && data.symptoms.length > 0) {

            output +=
                "<strong>Detected Symptoms:</strong><br>";

            data.symptoms.forEach(function(symptom) {

                output +=
                    "✓ " + symptom + "<br>";

            });

            output += "<br>";

        }


        output +=
            "<strong>Extracted Report Text:</strong><br><br>";

        output +=
            "<div class='report-text'>" +
            escapeHTML(data.report_text) +
            "</div>";


        resultText.innerHTML = output;

        result.classList.remove("hidden");

        result.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });


    } catch (error) {

        console.error(error);

        alert(
            "Unable to connect to HealthAI. " +
            "Make sure Flask is running."
        );

    } finally {

        loading.classList.add("hidden");

        button.disabled = false;

        button.innerText =
            "🧠 Analyze Medical Report";
    }
}


// ============================================================
// FILE SELECT
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function() {

        const fileInput =
            document.getElementById("reportFile");

        const message =
            document.getElementById("uploadMessage");


        if (fileInput) {

            fileInput.addEventListener(
                "change",
                function() {

                    if (fileInput.files.length) {

                        const file =
                            fileInput.files[0];

                        message.innerText =
                            "📄 " +
                            file.name +
                            " selected successfully.";

                    }

                }
            );

        }

    }
);


// ============================================================
// SAFE HTML
// ============================================================

function escapeHTML(text) {

    if (!text) {
        return "";
    }

    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}