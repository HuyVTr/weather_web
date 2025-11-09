// main.js
// Chứa toàn bộ logic JS cho tất cả các trang.

// --- LOGIC CHUNG (chạy trên mọi trang) ---
document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();

    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    
    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });
    }

    // Kiểm tra nếu là trang forecast thì khởi tạo
    if (document.getElementById('page-forecast')) {
        initForecastPage();
    }

    // Kiểm tra nếu là trang storm thì khởi tạo
    if (document.getElementById('page-storm')) {
        initStormMap();
    }
});

// --- LOGIC TRANG DỰ BÁO (/forecast) ---
function initForecastPage() {
    console.log("Khởi tạo trang Dự báo...");

    const dom = {
        provinceSelect: document.getElementById('province-select'),
        geolocateBtn: document.getElementById('geolocate-btn'),
        loadingSpinner: document.getElementById('loading-spinner'),
        currentWeather: document.getElementById('current-weather'),
        currentTemp: document.getElementById('current-temp'),
        currentHumidity: document.getElementById('current-humidity'),
        currentPressure: document.getElementById('current-pressure'),
        currentWind: document.getElementById('current-wind'),
        currentVisibility: document.getElementById('current-visibility'),
        currentUv: document.getElementById('current-uv'),
        dailyForecastContainer: document.getElementById('daily-forecast-container'),
        hourlyForecastContainer: document.getElementById('hourly-forecast-container'),
        aqiChart: document.getElementById('chart-aqi'),
    };

    let provincesData = [];

    async function loadProvinces() {
        try {
            const response = await fetch('/api/provinces');
            if (!response.ok) throw new Error('Không thể tải danh sách tỉnh');
            provincesData = await response.json();
            dom.provinceSelect.innerHTML = '<option value="">Chọn tỉnh...</option>';
            provincesData.forEach(province => {
                const option = document.createElement('option');
                option.value = province.name;
                option.textContent = province.name;
                dom.provinceSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Lỗi tải tỉnh:', error);
            alert('Lỗi khi tải danh sách tỉnh.');
        }
    }

    dom.geolocateBtn.addEventListener('click', handleGeolocation);
    dom.provinceSelect.addEventListener('change', (e) => fetchForecastData(e.target.value));

    // Load provinces và tự động lấy vị trí mặc định
    loadProvinces().then(() => handleGeolocation());

    async function handleGeolocation() {
        dom.loadingSpinner.classList.remove('hidden');
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(async (position) => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const geoResponse = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json&addressdetails=1`);
                const geoData = await geoResponse.json();
                const provinceName = geoData.address?.state || findNearestProvince(provincesData, lat, lon).name;
                dom.provinceSelect.value = provinceName;
                await fetchForecastData(provinceName);
            }, () => {
                alert('Không thể lấy vị trí. Sử dụng tỉnh mặc định (Hà Nội).');
                fetchForecastData('Hà Nội');
            });
        } else {
            alert('Trình duyệt không hỗ trợ vị trí. Sử dụng tỉnh mặc định (Hà Nội).');
            await fetchForecastData('Hà Nội');
        }
    }

    async function fetchForecastData(provinceName) {
        if (!provinceName) return;
        dom.loadingSpinner.classList.remove('hidden');
        dom.currentWeather.classList.add('hidden');
        dom.dailyForecastContainer.innerHTML = '';
        dom.hourlyForecastContainer.innerHTML = '';
        try {
            const response = await fetch(`/api/forecast?province=${encodeURIComponent(provinceName)}`);
            if (!response.ok) throw new Error('Không thể tải dữ liệu dự báo');
            const data = await response.json();
            renderForecast(data);
        } catch (error) {
            console.error('Lỗi tải dự báo:', error);
            alert('Lỗi khi tải dữ liệu dự báo.');
        } finally {
            dom.loadingSpinner.classList.add('hidden');
        }
    }

    function renderForecast(data) {
        // Render current weather
        dom.currentWeather.classList.remove('hidden');
        dom.currentTemp.textContent = data.current.temperature_2m;
        dom.currentHumidity.textContent = data.current.relative_humidity_2m;
        dom.currentPressure.textContent = data.current.pressure_msl;
        dom.currentWind.textContent = data.current.wind_speed_10m;
        dom.currentVisibility.textContent = (data.current.visibility / 1000).toFixed(1);
        dom.currentUv.textContent = data.current.uv_index;

        // Render hourly forecast as table
        dom.hourlyForecastContainer.innerHTML = '<table class="w-full text-center"><thead><tr><th>Giờ</th><th>Thời tiết</th><th>Nhiệt độ</th><th>Mưa</th></tr></thead><tbody></tbody></table>';
        const hourlyTableBody = dom.hourlyForecastContainer.querySelector('tbody');
        data.hourly.time.forEach((time, i) => {
            const hour = new Date(time).getHours();
            const weather = getWeatherDesc(data.hourly.weather_code[i]);
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${hour}:00</td>
                <td><i data-lucide="${weather.icon}"></i></td>
                <td>${data.hourly.temperature_2m[i]}°C</td>
                <td>${data.hourly.precipitation[i]} mm</td>
            `;
            hourlyTableBody.appendChild(row);
        });

        // Render 7-day forecast as table
        dom.dailyForecastContainer.innerHTML = '<table class="w-full text-left"><thead><tr><th>Ngày</th><th>Thời tiết</th><th>Mưa</th><th>Gió</th></tr></thead><tbody></tbody></table>';
        const dailyTableBody = dom.dailyForecastContainer.querySelector('tbody');
        data.daily.time.forEach((time, i) => {
            const date = new Date(time).toLocaleDateString('vi-VN');
            const weather = getWeatherDesc(data.daily.weather_code[i]);
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${date}</td>
                <td><i data-lucide="${weather.icon}"></i> ${weather.desc}</td>
                <td>${data.daily.precipitation_sum[i]} mm</td>
                <td>${data.daily.wind_speed_10m_max[i]} km/h</td>
            `;
            dailyTableBody.appendChild(row);
        });

        // Render AQI donut
        if (window.aqiChart) window.aqiChart.destroy();
        window.aqiChart = new Chart(dom.aqiChart, {
            type: 'doughnut',
            data: {
                labels: ['CO', 'NO2', 'O3', 'PM2.5', 'PM10', 'SO2'],
                datasets: [{
                    data: [
                        data.aqi.components.co || 0,
                        data.aqi.components.no2 || 0,
                        data.aqi.components.o3 || 0,
                        data.aqi.components.pm2_5 || 0,
                        data.aqi.components.pm10 || 0,
                        data.aqi.components.so2 || 0
                    ],
                    backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
                }]
            },
            options: {
                responsive: true,
                cutout: '70%',
                plugins: {
                    legend: { position: 'bottom' },
                    title: { display: true, text: 'Chất lượng không khí' }
                }
            }
        });
    }

    function getWeatherDesc(code) {
        const codes = {
            0: { desc: 'Trời quang', icon: 'sun' },
            1: { desc: 'Ít mây', icon: 'cloud' },
            2: { desc: 'Mây rải rác', icon: 'cloud' },
            3: { desc: 'Nhiều mây', icon: 'cloud' },
            45: { desc: 'Sương mù', icon: 'cloud-fog' },
            51: { desc: 'Mưa phùn nhẹ', icon: 'cloud-drizzle' },
            // Thêm các code khác từ Open-Meteo docs
            // ...
        };
        return codes[code] || { desc: 'Không rõ', icon: 'cloud-question' };
    }

    function findNearestProvince(provincesData, lat, lon) {
        let minDistance = Infinity;
        let nearestProvince = provincesData[0];
        provincesData.forEach(province => {
            const dLat = (province.lat - lat) * Math.PI / 180;
            const dLon = (province.lon - lon) * Math.PI / 180;
            const a = 0.5 - Math.cos(dLat) / 2 + Math.cos(lat * Math.PI / 180) * Math.cos(province.lat * Math.PI / 180) * (1 - Math.cos(dLon)) / 2;
            const distance = 12742 * Math.asin(Math.sqrt(a));
            if (distance < minDistance) { minDistance = distance; nearestProvince = province; }
        });
        return nearestProvince;
    }

    function chartOptions(y1ID, y1Label, y2ID = null, y2Label = null) {
        const options = {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { type: 'time', time: { unit: 'hour', tooltipFormat: 'HH:mm dd/MM' }, ticks: { color: '#9ca3af' }, grid: { color: '#374151' } },
                [y1ID]: { type: 'linear', position: 'left', title: { display: true, text: y1Label, color: '#e5e7eb' }, ticks: { color: '#9ca3af' }, grid: { color: '#374151' } }
            },
            plugins: { legend: { labels: { color: '#e5e7eb' } }, tooltip: { mode: 'index', intersect: false } }
        };
        if (y2ID) {
            options.scales[y2ID] = { type: 'linear', position: 'right', title: { display: true, text: y2Label, color: '#e5e7eb' }, ticks: { color: '#9ca3af' }, grid: { drawOnChartArea: false } };
        }
        return options;
    }
}

// --- LOGIC TRANG BÃO (/storm) ---
// Hàm này được gọi từ template storm.html
function initStormMap() {
    const map = L.map('map-container').setView([10.7769, 106.7009], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
    }).addTo(map);

    fetch('/api/storm_track')
        .then(res => res.json())
        .then(data => {
            if (data.no_storm) {
                L.marker([12.0, 109.0]).addTo(map).bindPopup(data.message).openPopup();
            } else {
                L.polyline(data.track.coordinates, {color: '#FADBD8', dashArray: '5, 5', weight: 3}).addTo(map);
                L.marker(data.center).addTo(map).bindPopup(`Tâm bão<br>Cảnh báo: ${data.warning}<br>Đi vào đất liền: ${data.landfall_vn ? 'Có' : 'Không'}`).openPopup();
            }
        })
        .catch(err => console.error(err));
}