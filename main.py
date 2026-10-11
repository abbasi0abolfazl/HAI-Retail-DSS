import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Data
raw_data_items = [
    {"id": "P01", "posSales": 120, "inventory": 25, "weather": "بارانی", "marginA": 0.28, "marginB": 0.12, "override": True, "humanVal": 5.0},
    {"id": "P02", "posSales": 40, "inventory": 85, "weather": "آفتابی", "marginA": 0.19, "marginB": 0.16, "override": False, "humanVal": 0.0},
    {"id": "P03", "posSales": 75, "inventory": 15, "weather": "آفتابی", "marginA": 0.32, "marginB": 0.1, "override": True, "humanVal": 7.5},
    {"id": "P04", "posSales": 95, "inventory": 55, "weather": "رویداد ویژه", "marginA": 0.22, "marginB": 0.24, "override": False, "humanVal": 0.0},
    {"id": "P05", "posSales": 15, "inventory": 90, "weather": "برفی", "marginA": 0.15, "marginB": 0.13, "override": False, "humanVal": 0.0}
]

class AppState:
    def __init__(self):
        self.discount = 0.95
        self.penalty = 0.5
        self.bias = 0.3
        self.tolerance = 0.05
        self.reward = 12.0
        self.trust = 0.88
        self.adapt = 0.2
        self.fuzzy = True
        self.xai = True
        self.biasOn = True
        self.human = True

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3.5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#fafafa')
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=2.5)
        
    def clear_chart(self):
        """Clear the chart completely"""
        self.ax.clear()
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=2.5)
        self.draw()
        
    def update_chart(self, labels, ai_actions, final_actions, rewards):
        # Clear and reset
        self.ax.clear()
        
        x = np.arange(len(labels))
        width = 0.3
        
        # Bar charts
        bars1 = self.ax.bar(x - width/2, ai_actions, width, 
                           label='AI Decision', 
                           color=(36/255, 54/255, 110/255, 0.7),
                           edgecolor='#1a2a5e', linewidth=1.2)
        
        bars2 = self.ax.bar(x + width/2, final_actions, width,
                           label='Final Decision',
                           color=(33/255, 150/255, 243/255, 0.7),
                           edgecolor='#1a73e8', linewidth=1.2)
        
        # Secondary axis for rewards
        ax2 = self.ax.twinx()
        ax2.plot(x, rewards, 'o-', color='#2d8f5e', 
                linewidth=2, markersize=6, label='Reward')
        
        # Styling
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, fontsize=10)
        self.ax.set_xlabel('Product', fontsize=10, fontweight='500')
        self.ax.set_ylabel('Price Change (%)', fontsize=10, fontweight='500')
        ax2.set_ylabel('Adjusted Reward', fontsize=10, fontweight='500', color='#2d8f5e')
        
        # Grid
        self.ax.grid(True, alpha=0.2, linestyle='--')
        
        # Legend
        lines1, labels1 = self.ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            self.ax.legend(lines1 + lines2, labels1 + labels2, 
                          loc='upper left', fontsize=8, framealpha=0.9)
        
        self.fig.tight_layout(pad=2.5)
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.app_state = AppState()
        # Initialize dictionaries before using them
        self.param_fields = {}
        self.ablation_toggles = {}
        self.stats_labels = {}
        self.formula_labels = {}
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("HAI-Retail DSS")
        self.setGeometry(100, 100, 1200, 800)
        
        # Modern style
        self.setStyleSheet("""
            QMainWindow {
                background: #f5f7fa;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #e0e4e8;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #1a2a5e;
            }
            QLabel {
                color: #2c3e50;
            }
            QDoubleSpinBox {
                padding: 5px;
                border-radius: 4px;
                border: 1px solid #d0d5dd;
                background: white;
                min-width: 80px;
            }
            QDoubleSpinBox:focus {
                border: 1px solid #4a90d9;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QTableWidget {
                gridline-color: #e8ecf0;
                background: white;
                alternate-background-color: #f8f9fa;
                border: 1px solid #e0e4e8;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QHeaderView::section {
                background: #e8ecf0;
                padding: 6px;
                font-weight: 600;
                border: none;
                color: #2c3e50;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f2f5;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #c0c8d4;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0aab8;
            }
            QScrollBar:horizontal {
                border: none;
                background: #f0f2f5;
                height: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #c0c8d4;
                border-radius: 5px;
                min-width: 20px;
            }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Sidebar
        sidebar = QWidget()
        sidebar.setMaximumWidth(350)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(10)
        
        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        
        param_configs = [
            ("discount", "Discount Factor (γ)", 0.01, 0, 1, 0.95),
            ("penalty", "Override Penalty (λ)", 0.05, 0, 10, 0.5),
            ("bias", "Bias Intensity (η)", 0.05, 0, 5, 0.3),
            ("tolerance", "Fairness Tolerance (δ)", 0.01, 0, 0.5, 0.05),
            ("reward", "Base Reward (R)", 0.5, 1, 50, 12.0),
            ("trust", "Trust Weight", 0.01, 0, 1, 0.88),
            ("adapt", "Adaptation Rate", 0.01, 0, 1, 0.2)
        ]
        
        for key, label, step, min_val, max_val, default in param_configs:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setMinimumWidth(150)
            row.addWidget(lbl)
            
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            spin.valueChanged.connect(lambda v, k=key: self.on_param_changed(k, v))
            row.addWidget(spin)
            params_layout.addLayout(row)
            self.param_fields[key] = spin
        
        sidebar_layout.addWidget(params_group)
        
        # Ablation
        ablation_group = QGroupBox("Ablation Study")
        ablation_layout = QVBoxLayout(ablation_group)
        
        ablation_configs = [
            ("fuzzy", "Fuzzy Logic Layer"),
            ("xai", "XAI Module"),
            ("biasOn", "Bias Control"),
            ("human", "Human-in-the-loop")
        ]
        
        for key, name in ablation_configs:
            row = QHBoxLayout()
            lbl = QLabel(name)
            row.addWidget(lbl)
            
            toggle = QCheckBox()
            toggle.setChecked(True)
            toggle.stateChanged.connect(lambda state, k=key: self.on_ablation_changed(k, state))
            row.addWidget(toggle)
            ablation_layout.addLayout(row)
            self.ablation_toggles[key] = toggle
        
        sidebar_layout.addWidget(ablation_group)
        
        # Stats
        stats_group = QGroupBox("System Stats")
        stats_layout = QGridLayout(stats_group)
        
        stats_configs = [
            ("avg_reward", "Avg Reward"),
            ("total_bias", "Total Bias"),
            ("human_count", "Human Interventions"),
            ("active_modules", "Active Modules")
        ]
        
        for i, (key, label) in enumerate(stats_configs):
            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight: 500; color: #5a6a7a;")
            val = QLabel("—")
            val.setStyleSheet("font-weight: 700; font-size: 13px; color: #1a2a5e;")
            stats_layout.addWidget(lbl, i, 0)
            stats_layout.addWidget(val, i, 1)
            self.stats_labels[key] = val
        
        sidebar_layout.addWidget(stats_group)
        sidebar_layout.addStretch()
        
        # Main content
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(10)
        
        # Chart
        chart_group = QGroupBox("Simulation Chart")
        chart_layout = QVBoxLayout(chart_group)
        self.canvas = MplCanvas(self, width=6, height=3.5)
        chart_layout.addWidget(self.canvas)
        content_layout.addWidget(chart_group)
        
        # Formulas
        formulas_group = QGroupBox("Formulas")
        formulas_layout = QVBoxLayout(formulas_group)
        
        formula_texts = [
            ("eq_actor", "Actor-Critic:"),
            ("eq_bias", "Bias Penalty:"),
            ("eq_reward", "Adjusted Reward:")
        ]
        
        for key, label in formula_texts:
            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight: 500; color: #5a6a7a;")
            formulas_layout.addWidget(lbl)
            val = QLabel("—")
            val.setStyleSheet("background: #f8f9fa; padding: 6px 10px; border-radius: 4px; font-family: monospace; border: 1px solid #e8ecf0;")
            val.setWordWrap(True)
            formulas_layout.addWidget(val)
            self.formula_labels[key] = val
        
        content_layout.addWidget(formulas_group)
        
        # Table
        table_group = QGroupBox("Data Table")
        table_layout = QVBoxLayout(table_group)
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Product", "POS", "Inventory", "AI Action", "Fuzzy",
            "XAI", "Intervention", "Final", "Reward"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.table)
        content_layout.addWidget(table_group)
        
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content)
        
        # Initial update
        self.update_ui()
    
    def on_param_changed(self, key, value):
        setattr(self.app_state, key, value)
        self.update_ui()
        
    def on_ablation_changed(self, key, state):
        setattr(self.app_state, key, state == Qt.Checked)
        self.update_ui()
    
    def process_simulation(self):
        results = []
        for row in raw_data_items:
            ai_action = 2.0
            if row["inventory"] < 30 and row["posSales"] > 70:
                ai_action = 15.0
            elif row["inventory"] > 80 and row["posSales"] < 50:
                ai_action = -10.0
            
            low_inv = max(0, min(1, (40 - row["inventory"]) / 20))
            fuzzy_action = ai_action
            fuzzy_status = "none"
            
            if self.app_state.fuzzy:
                if low_inv > 0.5 and ai_action > 10.0:
                    fuzzy_action = ai_action * 0.7
                    fuzzy_status = "adjusted"
            else:
                if low_inv > 0.5 and ai_action > 10.0:
                    fuzzy_status = "risk"
            
            xai_text = ""
            if self.app_state.xai:
                xai_text = f"POS:{row['posSales']} | Inv:{row['inventory']} | {row['weather']}"
            else:
                xai_text = "⚠️ XAI disabled"
            
            is_human = self.app_state.human and row["override"]
            final_action = fuzzy_action if not is_human else row["humanVal"]
            
            disparity = abs(row["marginA"] - row["marginB"])
            bias_penalty = 0
            if self.app_state.biasOn:
                bias_penalty = self.app_state.bias * max(0, disparity - self.app_state.tolerance)
            
            base_reward = self.app_state.reward * (0.96 if not self.app_state.xai else 1.0)
            adjusted_reward = (base_reward * self.app_state.trust) + (self.app_state.adapt * 5) - bias_penalty
            if is_human:
                adjusted_reward -= self.app_state.penalty
            
            results.append({
                "id": row["id"],
                "posSales": row["posSales"],
                "inventory": row["inventory"],
                "weather": row["weather"],
                "ai_action": ai_action,
                "fuzzy_action": fuzzy_action,
                "fuzzy_status": fuzzy_status,
                "xai_text": xai_text,
                "is_human": is_human,
                "final_action": final_action,
                "bias_penalty": bias_penalty,
                "adjusted_reward": adjusted_reward
            })
        return results
    
    def update_ui(self):
        results = self.process_simulation()
        
        # Stats
        avg_reward = sum(r["adjusted_reward"] for r in results) / len(results)
        total_bias = sum(r["bias_penalty"] for r in results)
        human_count = sum(1 for r in results if r["is_human"])
        active = sum([self.app_state.fuzzy, self.app_state.xai, self.app_state.biasOn, self.app_state.human])
        
        self.stats_labels["avg_reward"].setText(f"{avg_reward:.2f}")
        self.stats_labels["total_bias"].setText(f"{total_bias:.3f}")
        self.stats_labels["human_count"].setText(f"{human_count}/{len(results)}")
        self.stats_labels["active_modules"].setText(f"{active}/4")
        
        # Formulas
        mdp = f"V(s) = R(s) + {self.app_state.discount}·ΣP(s'|s,a)V(s')"
        bias_f = f"{self.app_state.bias}·max(0, |m_i-m_j|-{self.app_state.tolerance})" if self.app_state.biasOn else "0 (disabled)"
        reward_f = f"R_adj = {self.app_state.reward} - L_bias" + (f" - I·{self.app_state.penalty}" if self.app_state.human else "")
        
        self.formula_labels["eq_actor"].setText(mdp)
        self.formula_labels["eq_bias"].setText(bias_f)
        self.formula_labels["eq_reward"].setText(reward_f)
        
        # Table
        self.table.setRowCount(len(results))
        for i, r in enumerate(results):
            self.table.setItem(i, 0, QTableWidgetItem(r["id"]))
            self.table.setItem(i, 1, QTableWidgetItem(str(r["posSales"])))
            self.table.setItem(i, 2, QTableWidgetItem(str(r["inventory"])))
            self.table.setItem(i, 3, QTableWidgetItem(f"{r['ai_action']:.1f}%"))
            
            fuzzy_text = "None"
            if r["fuzzy_status"] == "adjusted":
                fuzzy_text = f"Adjusted\n{r['fuzzy_action']:.1f}%"
            elif r["fuzzy_status"] == "risk":
                fuzzy_text = f"Risk\n{r['fuzzy_action']:.1f}%"
            self.table.setItem(i, 4, QTableWidgetItem(fuzzy_text))
            
            self.table.setItem(i, 5, QTableWidgetItem(r["xai_text"]))
            self.table.setItem(i, 6, QTableWidgetItem("Human" if r["is_human"] else "AI"))
            self.table.setItem(i, 7, QTableWidgetItem(f"{r['final_action']:.1f}%"))
            
            reward_item = QTableWidgetItem(f"{r['adjusted_reward']:.2f}")
            reward_item.setForeground(QColor(40, 167, 69))
            reward_item.setFont(QFont("monospace", 9, QFont.Bold))
            self.table.setItem(i, 8, reward_item)
        
        self.table.resizeColumnsToContents()
        
        # Chart - clear and update
        labels = [r["id"] for r in results]
        ai_actions = [r["ai_action"] for r in results]
        final_actions = [r["final_action"] for r in results]
        rewards = [r["adjusted_reward"] for r in results]
        
        self.canvas.update_chart(labels, ai_actions, final_actions, rewards)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
