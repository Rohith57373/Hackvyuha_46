// Analytics page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log('Analytics page loaded');
    
    // Store chart objects for later updates
    let charts = {
        fpsChart: null,
        accuracyChart: null,
        fpsVsAccuracyChart: null,
        fpsSparkline: null,
        accuracySparkline: null,
        processingTimeSparkline: null,
        faceMatchesSparkline: null
    };
    
    // Storage for time-series data
    let timeSeriesData = {
        fps: Array(24).fill(0),
        accuracy: Array(24).fill(0),
        processingTime: Array(24).fill(0),
        faceMatches: Array(24).fill(0)
    };
    
    // Time labels for x-axis
    const timeLabels = Array.from({ length: 24 }, (_, i) => `${i}:00`);
    const dateLabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    
    // Generate random data between min and max with a step - used as fallback
    function generateRandomData(length, min, max, step = 0.1) {
        return Array.from({ length }, () => parseFloat((Math.random() * (max - min) + min).toFixed(1)));
    }
    
    // Generate slightly increasing random data between min and max - used as fallback
    function generateIncreasingData(length, min, max, maxStep = 2) {
        let data = [min + Math.random() * (max - min) * 0.3];
        for (let i = 1; i < length; i++) {
            const step = Math.random() * maxStep - maxStep * 0.3;
            const newValue = data[i - 1] + step;
            data.push(Math.min(Math.max(newValue, min), max));
        }
        return data;
    }
    
    // Initialize charts with placeholder data
    function initCharts() {
        // FPS sparkline chart
        charts.fpsSparkline = new Chart(document.getElementById('fpsSparkline'), {
            type: 'line',
            data: {
                labels: Array(12).fill(''),
                datasets: [{
                    data: generateRandomData(12, 25, 32),
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false,
                    }
                },
                scales: {
                    x: {
                        display: false,
                    },
                    y: {
                        display: false,
                    }
                },
                animation: {
                    duration: 1500
                }
            }
        });
        
        // Accuracy sparkline chart
        charts.accuracySparkline = new Chart(document.getElementById('accuracySparkline'), {
            type: 'line',
            data: {
                labels: Array(12).fill(''),
                datasets: [{
                    data: generateRandomData(12, 85, 98),
                    borderColor: '#10b981',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false,
                    }
                },
                scales: {
                    x: {
                        display: false,
                    },
                    y: {
                        display: false,
                    }
                },
                animation: {
                    duration: 1500
                }
            }
        });
        
        // Processing time sparkline
        charts.processingTimeSparkline = new Chart(document.getElementById('processingTimeSparkline'), {
            type: 'line',
            data: {
                labels: Array(12).fill(''),
                datasets: [{
                    data: generateRandomData(12, 20, 30),
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false,
                    }
                },
                scales: {
                    x: {
                        display: false,
                    },
                    y: {
                        display: false,
                    }
                },
                animation: {
                    duration: 1500
                }
            }
        });
        
        // Face matches sparkline
        charts.faceMatchesSparkline = new Chart(document.getElementById('faceMatchesSparkline'), {
            type: 'line',
            data: {
                labels: Array(12).fill(''),
                datasets: [{
                    data: [35, 42, 38, 45, 40, 50, 48, 58, 52, 48, 45, 60],
                    borderColor: '#6366f1',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false,
                    }
                },
                scales: {
                    x: {
                        display: false,
                    },
                    y: {
                        display: false,
                    }
                },
                animation: {
                    duration: 1500
                }
            }
        });
        
        // FPS Chart
        charts.fpsChart = new Chart(document.getElementById('fpsChart'), {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [{
                    label: 'FPS',
                    data: timeSeriesData.fps,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointBackgroundColor: '#3b82f6',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        align: 'end',
                        labels: {
                            font: {
                                size: 14
                            }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return `FPS: ${context.raw}`;
                            }
                        },
                        titleFont: {
                            size: 16
                        },
                        bodyFont: {
                            size: 14
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        suggestedMin: 20,
                        suggestedMax: 35,
                        title: {
                            display: true,
                            text: 'Frames Per Second',
                            font: {
                                size: 16,
                                weight: 'bold'
                            }
                        },
                        ticks: {
                            font: {
                                size: 14
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time (hours)',
                            font: {
                                size: 16,
                                weight: 'bold'
                            }
                        },
                        ticks: {
                            font: {
                                size: 14
                            }
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
                animation: {
                    duration: 1500
                }
            }
        });
        
        // Accuracy Chart
        charts.accuracyChart = new Chart(document.getElementById('accuracyChart'), {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [{
                    label: 'Accuracy',
                    data: timeSeriesData.accuracy,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointBackgroundColor: '#10b981',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        align: 'end',
                        labels: {
                            font: {
                                size: 14
                            }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return `Accuracy: ${context.raw}%`;
                            }
                        },
                        titleFont: {
                            size: 16
                        },
                        bodyFont: {
                            size: 14
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        suggestedMin: 80,
                        suggestedMax: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)',
                            font: {
                                size: 16,
                                weight: 'bold'
                            }
                        },
                        ticks: {
                            font: {
                                size: 14
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time (hours)',
                            font: {
                                size: 16,
                                weight: 'bold'
                            }
                        },
                        ticks: {
                            font: {
                                size: 14
                            }
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
                animation: {
                    duration: 1500
                }
            }
        });
        
        // FPS vs Accuracy comparison chart
        charts.fpsVsAccuracyChart = new Chart(document.getElementById('fpsVsAccuracyChart'), {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [
                    {
                        label: 'FPS',
                        data: timeSeriesData.fps,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0)',
                        borderWidth: 3,
                        pointRadius: 5,
                        pointBackgroundColor: '#3b82f6',
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Accuracy',
                        data: timeSeriesData.accuracy,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0)',
                        borderWidth: 3,
                        pointRadius: 5,
                        pointBackgroundColor: '#10b981',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        align: 'end',
                        labels: {
                            font: {
                                size: 14
                            }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        titleFont: {
                            size: 16
                        },
                        bodyFont: {
                            size: 14
                        }
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'FPS',
                            font: {
                                size: 16,
                                weight: 'bold'
                            }
                        },
                        suggestedMin: 20,
                        suggestedMax: 35,
                        ticks: {
                            font: {
                                size: 14
                            }
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Accuracy (%)',
                            font: {
                                size: 16,
                                weight: 'bold'
                            }
                        },
                        min: 80,
                        max: 100,
                        grid: {
                            drawOnChartArea: false
                        },
                        ticks: {
                            font: {
                                size: 14
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time (hours)',
                            font: {
                                size: 16,
                                weight: 'bold'
                            }
                        },
                        ticks: {
                            font: {
                                size: 14
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                animation: {
                    duration: 1500
                }
            }
        });
        
        // Face Detection Distribution
        const faceDetectionData = {
            chart: {
                type: 'donut',
                height: 400,
                fontFamily: 'Inter, sans-serif'
            },
            series: [55, 35, 10],
            labels: ['Recognized', 'Unrecognized', 'Low confidence'],
            colors: ['#10b981', '#f59e0b', '#ef4444'],
            legend: {
                position: 'bottom',
                fontSize: '14px',
                fontFamily: 'Inter, sans-serif',
                markers: {
                    width: 12,
                    height: 12
                },
                itemMargin: {
                    horizontal: 10,
                    vertical: 5
                }
            },
            plotOptions: {
                pie: {
                    donut: {
                        size: '55%',
                        labels: {
                            show: true,
                            name: {
                                show: true,
                                fontSize: '22px',
                                fontFamily: 'Inter, sans-serif',
                                fontWeight: 600,
                                color: undefined,
                                offsetY: -10
                            },
                            value: {
                                show: true,
                                fontSize: '18px',
                                fontFamily: 'Inter, sans-serif',
                                color: undefined,
                                offsetY: 16,
                                formatter: function (val) {
                                    return val + "%"
                                }
                            },
                            total: {
                                show: true,
                                showAlways: false,
                                label: 'Total',
                                fontSize: '22px',
                                fontFamily: 'Inter, sans-serif',
                                fontWeight: 600,
                                color: '#373d3f',
                                formatter: function (w) {
                                    return w.globals.seriesTotals.reduce((a, b) => {
                                        return a + b
                                    }, 0) + "%"
                                }
                            }
                        }
                    }
                }
            }
        };
        
        var faceDetectionChart = new ApexCharts(document.getElementById('faceDetectionChart'), faceDetectionData);
        faceDetectionChart.render();
    }
    
    // Fetch metrics from the API
    async function fetchMetrics() {
        try {
            const response = await fetch('/api/analytics/metrics');
            const data = await response.json();
            
            if (data.success) {
                return data.metrics;
            } else {
                console.error('Error fetching metrics:', data.error);
                return null;
            }
        } catch (error) {
            console.error('Failed to fetch metrics:', error);
            return null;
        }
    }
    
    // Update charts with real data
    function updateCharts(metrics) {
        if (!metrics) return;
        
        // Update overall metrics display
        const overall = metrics.overall;
        document.getElementById('avgFps').innerText = overall.avg_fps;
        document.getElementById('detectionAccuracy').innerText = overall.avg_accuracy + '%';
        
        // Processing time is average across cameras
        let avgProcessingTime = 0;
        let cameraCount = 0;
        for (const cameraId in metrics) {
            if (cameraId !== 'overall') {
                avgProcessingTime += metrics[cameraId].processing_time;
                cameraCount++;
            }
        }
        avgProcessingTime = cameraCount > 0 ? Math.round(avgProcessingTime / cameraCount) : 25;
        document.getElementById('processingTime').innerText = avgProcessingTime + 'ms';
        
        // Update face matches metric
        document.getElementById('faceMatches').innerText = overall.total_matches;
        
        // Shift time series data and add new point
        timeSeriesData.fps.shift();
        timeSeriesData.fps.push(overall.avg_fps);
        
        timeSeriesData.accuracy.shift();
        timeSeriesData.accuracy.push(overall.avg_accuracy);
        
        timeSeriesData.processingTime.shift();
        timeSeriesData.processingTime.push(avgProcessingTime);
        
        timeSeriesData.faceMatches.shift();
        timeSeriesData.faceMatches.push(overall.total_matches % 100); // Just to keep it in reasonable range
        
        // Update line charts
        updateLineCharts();
        
        // Update sparkline charts
        updateSparklines();
        
        // Update camera comparison table
        updateCameraTable(metrics);
    }
    
    // Update the main line charts
    function updateLineCharts() {
        if (charts.fpsChart) {
            charts.fpsChart.data.datasets[0].data = timeSeriesData.fps;
            charts.fpsChart.update();
        }
        
        if (charts.accuracyChart) {
            charts.accuracyChart.data.datasets[0].data = timeSeriesData.accuracy;
            charts.accuracyChart.update();
        }
        
        if (charts.fpsVsAccuracyChart) {
            charts.fpsVsAccuracyChart.data.datasets[0].data = timeSeriesData.fps;
            charts.fpsVsAccuracyChart.data.datasets[1].data = timeSeriesData.accuracy;
            charts.fpsVsAccuracyChart.update();
        }
    }
    
    // Update sparkline mini charts
    function updateSparklines() {
        // Get the last 12 points for sparklines
        const fpsData = timeSeriesData.fps.slice(-12);
        const accuracyData = timeSeriesData.accuracy.slice(-12);
        const processingTimeData = timeSeriesData.processingTime.slice(-12);
        const faceMatchesData = timeSeriesData.faceMatches.slice(-12);
        
        if (charts.fpsSparkline) {
            charts.fpsSparkline.data.datasets[0].data = fpsData;
            charts.fpsSparkline.update();
        }
        
        if (charts.accuracySparkline) {
            charts.accuracySparkline.data.datasets[0].data = accuracyData;
            charts.accuracySparkline.update();
        }
        
        if (charts.processingTimeSparkline) {
            charts.processingTimeSparkline.data.datasets[0].data = processingTimeData;
            charts.processingTimeSparkline.update();
        }
        
        if (charts.faceMatchesSparkline) {
            charts.faceMatchesSparkline.data.datasets[0].data = faceMatchesData;
            charts.faceMatchesSparkline.update();
        }
    }
    
    // Update camera comparison table
    function updateCameraTable(metrics) {
        const tableBody = document.querySelector('#cameraComparisonTable tbody');
        if (!tableBody) return;
        
        // Clear existing rows
        tableBody.innerHTML = '';
        
        // Add camera rows
        for (const cameraId in metrics) {
            if (cameraId === 'overall') continue;
            
            const camera = metrics[cameraId];
            const row = document.createElement('tr');
            
            // Apply alternating row colors
            row.className = 'bg-white';
            
            // Camera name
            const nameCell = document.createElement('td');
            nameCell.className = 'px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-800';
            nameCell.textContent = camera.name;
            row.appendChild(nameCell);
            
            // FPS
            const fpsCell = document.createElement('td');
            fpsCell.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-500';
            fpsCell.textContent = camera.processing_fps;
            row.appendChild(fpsCell);
            
            // Accuracy
            const accuracyCell = document.createElement('td');
            accuracyCell.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-500';
            accuracyCell.textContent = camera.accuracy + '%';
            row.appendChild(accuracyCell);
            
            // Faces Detected
            const facesCell = document.createElement('td');
            facesCell.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-500';
            facesCell.textContent = camera.detections;
            row.appendChild(facesCell);
            
            // Processing Time
            const timeCell = document.createElement('td');
            timeCell.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-500';
            timeCell.textContent = camera.processing_time + 'ms';
            row.appendChild(timeCell);
            
            // Status
            const statusCell = document.createElement('td');
            statusCell.className = 'px-6 py-4 whitespace-nowrap text-sm';
            
            const statusBadge = document.createElement('span');
            if (camera.processing_fps > 25) {
                statusBadge.className = 'px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800';
                statusBadge.textContent = 'Excellent';
            } else if (camera.processing_fps > 20) {
                statusBadge.className = 'px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800';
                statusBadge.textContent = 'Good';
            } else if (camera.processing_fps > 15) {
                statusBadge.className = 'px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-yellow-100 text-yellow-800';
                statusBadge.textContent = 'Fair';
            } else {
                statusBadge.className = 'px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800';
                statusBadge.textContent = 'Poor';
            }
            
            statusCell.appendChild(statusBadge);
            row.appendChild(statusCell);
            
            tableBody.appendChild(row);
        }
    }
    
    // Initialize and set update interval
    function init() {
        // First, initialize charts with placeholder data
        initCharts();
        
        // Then fetch initial data
        fetchMetrics().then(metrics => {
            if (metrics) {
                // Initialize time series data with real values
                const fpsValue = metrics.overall.avg_fps;
                const accuracyValue = metrics.overall.avg_accuracy;
                
                // Fill time series with slightly varying data based on initial values
                timeSeriesData.fps = Array.from({ length: 24 }, () => fpsValue + (Math.random() * 4 - 2));
                timeSeriesData.accuracy = Array.from({ length: 24 }, () => 
                    Math.min(99.9, Math.max(85, accuracyValue + (Math.random() * 3 - 1.5)))
                );
                
                // Initial update of charts and table
                updateCharts(metrics);
            }
        });
        
        // Set up periodic updates
        setInterval(async () => {
            const metrics = await fetchMetrics();
            if (metrics) {
                updateCharts(metrics);
            }
        }, 3000); // Update every 3 seconds
    }
    
    // Start the application
    init();
}); 