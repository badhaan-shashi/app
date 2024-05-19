$(document).ready(function() {
    // Set the source of the image
    $('#imageDisplay').attr('src', 'your_image_url.jpg').on('load', function() {
        const width = this.naturalWidth;
        const height = this.naturalHeight;
        generateAxisLabels(width, height);
    });
    // git commit
    function generateAxisLabels(width, height) {
        const xAxisContainer = $('.x-axis-labels');
        const yAxisContainer = $('.y-axis-labels');

        // Generate labels based on dimensions
        for (let i = 0; i <= width; i += 100) { // Adjust step as needed
            xAxisContainer.append(`<span>${i}</span> `);
        }
        for (let i = 0; i <= height; i += 100) { // Adjust step as needed
            yAxisContainer.append(`<span>${i}</span><br>`);
        }
    }
});