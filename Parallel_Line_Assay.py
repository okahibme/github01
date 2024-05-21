from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def transform_response(response, method):
    if method == 'log':
        response = np.clip(response, 1e-10, None)
        return np.log10(response)
    elif method == 'sqrt':
        return np.sqrt(response)
    elif method == 'boxcox':
        response = np.clip(response, 1e-10, None)
        transformed_response, _ = stats.boxcox(response)
        return transformed_response
    elif method == 'logit':
        response = np.clip(response, 1e-10, 1 - 1e-10)  # Avoiding 0 and 1 for logit
        return np.log(response / (1 - response))
    return response

def convert_nan_to_none(obj):
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_nan_to_none(i) for i in obj]
    return obj

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json.get('data')
        transformation = request.json.get('transformation')
        if not data:
            raise ValueError("No data provided")

        app.logger.info(f"Received data: {data}")
        app.logger.info(f"Transformation method: {transformation}")

        df = pd.DataFrame(data)
        df['dose'] = df['dose'].astype(float)
        df['response'] = df['response'].astype(float)
        df['group'] = df['group'].astype(int)

        app.logger.info(f"DataFrame: {df}")

        # 常用対数変換
        df['log_dose'] = np.log10(df['dose'])
        app.logger.info(f"Log-transformed DataFrame: {df}")

        # Responseの変換
        df['transformed_response'] = transform_response(df['response'], transformation)
        app.logger.info(f"Transformed Response DataFrame: {df[['response', 'transformed_response']]}")

        # Seabornのスタイルとカラーパレットを設定
        sns.set_style("darkgrid")  # スタイルをdarkgridに変更
        sns.set_palette(sns.color_palette(["blue", "orange"]))  # カラーパレットを変更

        # グラフの生成
        plots = []

        # Plot 1: 横軸=dose, 縦軸=response
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x='dose', y='response', hue='group', data=df, ax=ax1, palette=["blue", "orange"])
        ax1.set_title('Dose vs Response')
        img1 = io.BytesIO()
        plt.savefig(img1, format='png')
        img1.seek(0)
        plots.append(base64.b64encode(img1.getvalue()).decode())

        # Plot 2: 横軸=logdose, 縦軸=response
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x='log_dose', y='response', hue='group', data=df, ax=ax2, palette=["blue", "orange"])
        ax2.set_title('Log Dose vs Response')
        img2 = io.BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        plots.append(base64.b64encode(img2.getvalue()).decode())

        # Plot 3: 横軸=logdose, 縦軸=transformed response
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x='log_dose', y='transformed_response', hue='group', data=df, ax=ax3, palette=["blue", "orange"])
        ax3.set_title('Log Dose vs Transformed Response')
        img3 = io.BytesIO()
        plt.savefig(img3, format='png')
        img3.seek(0)
        plots.append(base64.b64encode(img3.getvalue()).decode())

        # ダミー変数の追加
        df['sample'] = df['group'].apply(lambda x: 1 if x == 2 else 0)

        # 交互作用項を含むモデルの適合
        df['interaction'] = df['log_dose'] * df['sample']
        X_interaction = df[['log_dose', 'sample', 'interaction']]
        X_interaction = sm.add_constant(X_interaction)
        y = df['transformed_response']
        model_interaction = sm.OLS(y, X_interaction).fit()

        app.logger.info(f"Interaction Model params: {model_interaction.params}")

        # 交互作用項の有意性の検定
        p_value_interaction = model_interaction.pvalues['interaction']
        are_lines_parallel = bool(p_value_interaction > 0.05)

        app.logger.info(f"Interaction p-value: {p_value_interaction}, Are lines parallel: {are_lines_parallel}")

        # 交互作用項なしのモデルに適合
        X = df[['log_dose', 'sample']]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        app.logger.info(f"Non-interaction Model params: {model.params}")

        # パラメータの取得
        intercept = model.params['const']
        slope = model.params['log_dose']
        delta = model.params['sample']

        app.logger.info(f"Intercept: {intercept}, Slope: {slope}, Delta: {delta}")

        # 相対力価の計算
        relative_potency = 10 ** (delta / slope)  # 常用対数に変更

        app.logger.info(f"Relative potency: {relative_potency}")

        # 信頼区間の計算
        conf = model.conf_int(alpha=0.05)
        conf_dict = {
            'const': [float(conf.loc['const'][0]), float(conf.loc['const'][1])],
            'log_dose': [float(conf.loc['log_dose'][0]), float(conf.loc['log_dose'][1])],
            'sample': [float(conf.loc['sample'][0]), float(conf.loc['sample'][1])]
        }

        # NaN を None に変換
        conf_dict = convert_nan_to_none(conf_dict)

        app.logger.info(f"Confidence intervals: {conf_dict}")

        # Relative Potencyの信頼区間の計算
        cov_matrix = model.cov_params()
        delta_var = cov_matrix.loc['sample', 'sample']
        slope_var = cov_matrix.loc['log_dose', 'log_dose']
        cov_delta_slope = cov_matrix.loc['sample', 'log_dose']

        # デルタ法による信頼区間の計算
        rp_variance = (delta / slope) ** 2 * (delta_var / delta ** 2 + slope_var / slope ** 2 - 2 * cov_delta_slope / (delta * slope))
        rp_se = np.sqrt(rp_variance)
        z_score = stats.norm.ppf(1 - 0.05 / 2)

        rp_conf_lower = relative_potency * np.exp(-z_score * rp_se)
        rp_conf_upper = relative_potency * np.exp(z_score * rp_se)

        app.logger.info(f"Relative potency confidence interval: [{rp_conf_lower}, {rp_conf_upper}]")

        # グラフの更新（ラインを引く）
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x='log_dose', y='transformed_response', hue='group', data=df, ax=ax3, palette=["blue", "orange"])
        doses = np.linspace(df['log_dose'].min(), df['log_dose'].max(), 100)
        group1_line = intercept + slope * doses
        group2_line = intercept + delta + slope * doses
        sns.lineplot(x=doses, y=group1_line, ax=ax3, color='blue', linestyle='--')
        sns.lineplot(x=doses, y=group2_line, ax=ax3, color='orange', linestyle='--')
        ax3.set_title('Log Dose vs Transformed Response (with fit lines)')
        img3 = io.BytesIO()
        plt.savefig(img3, format='png')
        img3.seek(0)
        plots[2] = base64.b64encode(img3.getvalue()).decode()

        # 残渣プロットの生成
        residuals = y - model.fittedvalues
        fig4, ax4 = plt.subplots(figsize=(8, 4))  # サイズを小さくする
        sns.scatterplot(x=df['log_dose'], y=residuals, hue=df['group'], ax=ax4, palette=["blue", "orange"])
        ax4.axhline(0, ls='--', c='gray')
        ax4.set_title('Residuals Plot')
        img4 = io.BytesIO()
        plt.savefig(img4, format='png')
        img4.seek(0)
        residuals_plot = base64.b64encode(img4.getvalue()).decode()

        # 注釈の追加
        relative_potency_note = f"Relative potency is a reference value due to significant non-parallelism (p = {p_value_interaction:.4f})" if not are_lines_parallel else f"Lines are parallel (p = {p_value_interaction:.4f})"

        result = {
            "intercept": round(float(intercept), 3),
            "slope": round(float(slope), 3),
            "delta": round(float(delta), 3),
            "relative_potency": round(float(relative_potency), 3),
            "relative_potency_confidence_interval": [round(rp_conf_lower, 3), round(rp_conf_upper, 3)],
            "confidence_intervals": {
                'const': [round(conf_dict['const'][0], 3), round(conf_dict['const'][1], 3)],
                'log_dose': [round(conf_dict['log_dose'][0], 3), round(conf_dict['log_dose'][1], 3)],
                'sample': [round(conf_dict['sample'][0], 3), round(conf_dict['sample'][1], 3)]
            },
            "plots": plots,
            "are_lines_parallel": are_lines_parallel,
            "p_value_interaction": round(float(p_value_interaction), 3),
            "relative_potency_note": relative_potency_note,
            "residuals_plot": residuals_plot
        }

        app.logger.info(f"Result: {result}")

        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
