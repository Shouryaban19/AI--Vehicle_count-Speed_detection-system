let video = document.getElementById("video");

document.getElementById("uploadForm").onsubmit = async (e) => {
    e.preventDefault();

    let formData = new FormData(e.target);

    await fetch("/upload", {
        method: "POST",
        body: formData
    });

    alert("Video Uploaded. Click Start.");
};

// START BUTTON
document.getElementById("startBtn").onclick = () => {
    video.src = "/video_feed";
};

// STOP BUTTON
document.getElementById("stopBtn").onclick = () => {
    video.src = "";
};

// LIVE COUNT STATS
setInterval(async () => {
    let res = await fetch("/stats");
    let data = await res.json();

    document.getElementById("stats").innerHTML =
        `Total: ${data.total}<br>
         Cars: ${data.car}<br>
         Buses: ${data.bus}<br>
         Trucks: ${data.truck}<br>
         Bikes: ${data.bike}`;
}, 1000);

// LIVE SPEED TABLE
setInterval(async () => {
    let res = await fetch("/speed_data");
    let data = await res.json();

    let table = "";

    data.forEach(d => {
        let rowClass = d.overspeed ? "overspeed" : "";

        table += `<tr class="${rowClass}">
                    <td>${d.id}</td>
                    <td>${d.speed}</td>
                  </tr>`;
    });

    document.getElementById("speedTable").innerHTML = table;

}, 1000);