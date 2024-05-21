$(document).ready(function() {
    // DataTablesの初期化
    const table1 = $('#group1-table').DataTable({
        paging: false,
        searching: false,
        info: false
    });

    const table2 = $('#group2-table').DataTable({
        paging: false,
        searching: false,
        info: false
    });

    // 行を追加するボタンのイベントリスナー
    $('#add-row-group1').on('click', function() {
        table1.row.add([
            '<input type="text" class="dose" />',
            '<input type="text" class="response" />'
        ]).draw();
    });

    $('#add-row-group2').on('click', function() {
        table2.row.add([
            '<input type="text" class="dose" />',
            '<input type="text" class="response" />'
        ]).draw();
    });

    // データ送信関数
    function submitData(transformation) {
        const group1Data = table1.rows().nodes().to$().map(function() {
            const row = $(this);
            return {
                dose: parseFloat(row.find('input.dose').val()),
                response: parseFloat(row.find('input.response').val()),
                group: 1
            };
        }).get();

        const group2Data = table2.rows().nodes().to$().map(function() {
            const row = $(this);
            return {
                dose: parseFloat(row.find('input.dose').val()),
                response: parseFloat(row.find('input.response').val()),
                group: 2
            };
        }).get();

        const data = [...group1Data, ...group2Data];

        console.log("Data to be sent:", data);
        console.log("Transformation:", transformation);

        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ data, transformation }),
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error);
                });
            }
            return response.json();
        })
        .then(result => {
            console.log("Result received:", result);
            const resultDiv = document.getElementById('result');
            if (result.error) {
                resultDiv.innerHTML = `Error: ${result.error}`;
            } else {
                resultDiv.innerHTML = `
                    Intercept: ${result.intercept}<br>
                    Slope: ${result.slope}<br>
                    Delta: ${result.delta}<br>
                    Relative Potency: ${result.relative_potency}<br>
                    Relative Potency Confidence Interval: [${result.relative_potency_confidence_interval[0]}, ${result.relative_potency_confidence_interval[1]}]<br>
                    Confidence Intervals: <br>
                    const: [${result.confidence_intervals.const[0]}, ${result.confidence_intervals.const[1]}]<br>
                    log_dose: [${result.confidence_intervals.log_dose[0]}, ${result.confidence_intervals.log_dose[1]}]<br>
                    sample: [${result.confidence_intervals.sample[0]}, ${result.confidence_intervals.sample[1]}]<br>
                    ${result.relative_potency_note}<br>
                `;
                const plotsDiv = document.getElementById('plots');
                plotsDiv.innerHTML = '';
                result.plots.forEach(plot => {
                    const img = document.createElement('img');
                    img.src = 'data:image/png;base64,' + plot;
                    plotsDiv.appendChild(img);
                });
                const residualsPlotDiv = document.getElementById('residuals-plot');
                residualsPlotDiv.innerHTML = '';
                if (result.residuals_plot) {
                    const img = document.createElement('img');
                    img.src = 'data:image/png;base64,' + result.residuals_plot;
                    residualsPlotDiv.appendChild(img);
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = `Error: ${error.message}`;
        });
    }

    // Submitボタンのイベントリスナー
    $('#submit-btn').on('click', function() {
        const transformation = document.querySelector('input[name="transformation"]:checked').value;
        submitData(transformation);
    });
});
