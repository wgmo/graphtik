function graphtik_armPanZoom() {
    for (const obj_el of document.querySelectorAll(".graphtik-zoomable-svg")) {
        svg_el = obj_el.contentDocument.querySelector("svg")
        var zoom_opts = "graphtikSvgZoomOptions" in obj_el.dataset ?
            JSON.parse(obj_el.dataset.graphtikSvgZoomOptions) : {};
        svgPanZoom(svg_el, zoom_opts);
    };
};

window.addEventListener("load", graphtik_armPanZoom);
