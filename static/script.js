document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('predictionForm');
    const venueSelect = document.getElementById('venue');
    const battingTeamSelect = document.getElementById('batting_team');
    const bowlingTeamSelect = document.getElementById('bowling_team');

    // Result elements
    const placeholder = document.getElementById('placeholder');
    const loader = document.getElementById('loader');
    const resultCard = document.getElementById('resultCard');
    const predictedValue = document.getElementById('predictedValue');
    const winProbContainer = document.getElementById('winProbContainer');
    const winProbValue = document.getElementById('winProbValue');

    let myChart = null;

    // Fetch metadata
    fetch('/get_metadata')
        .then(response => response.json())
        .then(data => {
            data.venues.forEach(venue => {
                const option = new Option(venue, venue);
                venueSelect.add(option);
            });
            data.teams.forEach(team => {
                const opt1 = new Option(team, team);
                const opt2 = new Option(team, team);
                battingTeamSelect.add(opt1);
                bowlingTeamSelect.add(opt2);
            });
        });

    form.addEventListener('submit', function (e) {
        e.preventDefault();

        const data = {
            venue: venueSelect.value,
            batting_team: battingTeamSelect.value,
            bowling_team: bowlingTeamSelect.value,
            current_score: document.getElementById('current_score').value,
            wickets: document.getElementById('wickets').value,
            overs: document.getElementById('overs').value,
            target_score: document.getElementById('target_score').value
        };

        placeholder.classList.add('hidden');
        resultCard.classList.add('hidden');
        loader.classList.remove('hidden');

        fetch('/predict_score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        })
            .then(response => response.json())
            .then(res => {
                loader.classList.add('hidden');
                if (res.status === 'success') {
                    resultCard.classList.remove('hidden');
                    showScoreResult(res.prediction, res.win_probability);
                } else {
                    alert('Error: ' + res.error);
                    placeholder.classList.remove('hidden');
                }
            })
            .catch(err => {
                loader.classList.add('hidden');
                alert('Something went wrong!');
                placeholder.classList.remove('hidden');
            });
    });

    function showScoreResult(score, winProb) {
        predictedValue.textContent = score;

        if (winProb !== null && winProb !== undefined) {
            winProbContainer.classList.remove('hidden');
            winProbValue.textContent = winProb + '%';
        } else {
            winProbContainer.classList.add('hidden');
        }

        updateChart(score);
    }

    function updateChart(score) {
        const ctx = document.getElementById('predictionChart').getContext('2d');
        if (myChart) myChart.destroy();

        myChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Min Range', 'Predicted', 'Max Range'],
                datasets: [{
                    label: 'Runs',
                    data: [score - 15, score, score + 15],
                    backgroundColor: ['rgba(14, 165, 233, 0.4)', 'rgba(34, 211, 238, 0.8)', 'rgba(14, 165, 233, 0.4)'],
                    borderColor: ['#0ea5e9', '#22d3ee', '#0ea5e9'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: false, ticks: { color: '#94a3b8' } }, x: { ticks: { color: '#94a3b8' } } },
                plugins: { legend: { display: false } }
            }
        });
    }
});
