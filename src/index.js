const loadRenderer = async (type) => {
    if (type === "probe") {
        const { default: runProbe } = await import("./probe-volume.js");
        runProbe({
            drawProbes: document.getElementById("drawProbes").checked,
            probeDensity: parseInt(document.getElementById("probeDensity").value),
            probeSamples: parseInt(document.getElementById("probeSamples").value),
        });
    } else {
        const { default: runPathTracer } = await import("./volume-pathtracer.js");
        runPathTracer();
    }
};

rendererSelect.addEventListener("change", () => {
    // Save the selected renderer before reload
    localStorage.setItem("selectedRenderer", rendererSelect.value);
    location.reload();
});

document.addEventListener("DOMContentLoaded", () => {
    const rendererSelect = document.getElementById("rendererSelect");
    const probeControls = document.getElementById("probeControls");

    // Restore renderer from localStorage
    const savedRenderer = localStorage.getItem("selectedRenderer");
    if (savedRenderer) {
        rendererSelect.value = savedRenderer;
    }

    const updateUI = () => {
        const renderer = rendererSelect.value;
        probeControls.style.display = renderer === "probe" ? "block" : "none";
    };

    updateUI();

    // Load the selected renderer
    loadRenderer(rendererSelect.value);
});
