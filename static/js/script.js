function showTooltip(event, craters) {
    const tooltip = document.getElementById('tooltip');
    const img = event.target;
    const rect = img.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    // Hardcode expected image size (500x500) for testing
    const scaleX = 500 / rect.width;
    const scaleY = 500 / rect.height;

    console.log(`Mouse position: (${x}, ${y}), Scale: (${scaleX}, ${scaleY})`);
    console.log('Craters:', craters);

    let tooltipText = '';
    let inCrater = false;
    let craterCenterX, craterCenterY;

    for (let crater of craters) {
        const scaledX = crater.center_x / scaleX;
        const scaledY = crater.center_y / scaleY;
        const scaledRadius = crater.radius / scaleX;
        const dist = Math.sqrt((x - scaledX) ** 2 + (y - scaledY) ** 2);
        console.log(`Crater: (${scaledX.toFixed(2)}, ${scaledY.toFixed(2)}), Radius: ${scaledRadius.toFixed(2)}, Distance: ${dist.toFixed(2)}`);
        if (dist <= scaledRadius * 1.5) {
            tooltipText = `Diameter: ${crater.diameter_m.toFixed(2)}m<br>Depth: ${crater.depth_m.toFixed(2)}m<br>Volume: ${crater.volume_m3.toExponential(2)}m³`;
            inCrater = true;
            craterCenterX = scaledX;
            craterCenterY = scaledY;
            console.log('Tooltip triggered for crater');
            break;
        }
    }

    if (inCrater) {
        tooltip.style.display = 'block';
        const tooltipX = Math.min(rect.width - 150, Math.max(0, craterCenterX + 10));
        const tooltipY = Math.min(rect.height - 50, Math.max(0, craterCenterY - 40));
        tooltip.style.left = (rect.left + tooltipX) + 'px';
        tooltip.style.top = (rect.top + tooltipY) + 'px';
        tooltip.innerHTML = tooltipText;
    } else {
        tooltip.style.display = 'none';
    }
}

function hideTooltip() {
    document.getElementById('tooltip').style.display = 'none';
}

function makeChartInteractive(chartId, data) {
    // Placeholder for chart interactivity (using Chart.js in future if needed)
}