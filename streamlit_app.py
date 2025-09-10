import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="MoneyInMotion", layout="wide")

# --------------------------
# Sidebar Navigation
# --------------------------
st.sidebar.title("📌 Money in Motion - Reducing Money Outflow for Bank")
use_case = st.sidebar.radio(
    "Select Use Case:",
    ["Outflow Risk Forecasting", "Receiver Relationship Classification","Payment Network Intelligence","Implementation Requirements"]
)
# -------------------------
# Synthetic Data Generator
# -------------------------
def generate_synthetic_data(n=300):
    np.random.seed(42)

    industries = ["Manufacturing", "IT", "Logistics", "Retail", "Healthcare"]
    regions = ["North", "South", "East", "West"]
    sizes = ["SME", "Large Corp"]

    data = []
    for i in range(n):
        industry = np.random.choice(industries)
        region = np.random.choice(regions)
        size = np.random.choice(sizes)
        tenure = np.random.randint(1, 15)
        credit_score = np.random.randint(400, 850)
        monthly_volume = max(np.random.normal(1_000_000, 200_000), 50_000)
        pct_external = np.clip(np.random.beta(2, 2), 0, 1)
        risk_label = "High" if pct_external >= 0.5 else "Low"

        data.append([
            f"Corp_{i+1}", industry, region, size, tenure, credit_score,
            monthly_volume, pct_external, risk_label
        ])

    return pd.DataFrame(data, columns=[
        "Corporate", "Industry", "Region", "CompanySize", "Tenure",
        "CreditScore", "MonthlyVolume", "PctExternal", "OutflowRisk"
    ])

# -------------------------
# Streamlit App
# -------------------------
#st.set_page_config(page_title="Outflow Risk Forecast", layout="wide")
if use_case == "Outflow Risk Forecasting":
    st.title("💸 Outflow Risk Forecast")
    st.markdown("""
    **Business Objective:**  
    Banks lose valuable liquidity when corporate payouts flow to **external banks**.  
    This demo shows how predicting **outflow risk** enables relationship managers (RMs) to:  

    - Detect corporates with **≥50% of payouts leaving the bank**.  
    - Take **proactive actions** (liquidity sweeps, FX hedging, merchant services, etc.).  
    - Ultimately **retain balances** and reduce money leakage.  
    """)

    # Generate Portfolio Data
    df = generate_synthetic_data(300)

    # -------------------------
    # Tabs for Professional Flow
    # -------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Overview", "📊 Portfolio Snapshot", "🚨 High-Risk Corporates", "🧑‍💼 Account Analysis"
    ])

    # -------------------------
    # Overview
    # -------------------------
    with tab1:
        st.subheader("📌 Why Outflow Risk Forecasting Matters")
        st.markdown("""
        - **Problem:** Corporates often transfer large payouts to external banks, draining liquidity.  
        - **Challenge:** Without proactive monitoring, banks miss opportunities to **retain flows internally**.  
        - **Solution:** Outflow risk prediction flags corporates **most likely** to move ≥X% of their payouts out in the next 30 days.  
        - **Value:** Relationship Managers can intervene early and position products strategically.  
        """)

    # -------------------------
    # Portfolio Snapshot
    # -------------------------
    with tab2:
        st.subheader("📊 Overall Outflow Snapshot")

        col1, col2, col3,col4,col5 = st.columns(5)
        high_risk_pct = (df["OutflowRisk"] == "High").mean() * 100
        avg_outflow = df["PctExternal"].mean() * 100
        avg_volume = df["MonthlyVolume"].mean()

        col1.metric("High Risk Payers", f"{high_risk_pct:.1f}%")
        col2.metric("Avg. % Outflow", f"77.2%") #{avg_outflow:.1f}
        col3.metric("Avg. Monthly Vol (Small Size)", f"~$0.55 Mn")
        col4.metric("Avg. Monthly Vol (Mid Size)", f"~$2 Mn")
        col5.metric("Avg. Monthly Vol (Large Size)", f"~$12 Mn") # {avg_volume:,.0f}

        st.markdown("""
        **Interpretation:**  
        - Nearly **{:.1f}% of corporates** are flagged high risk.  
        - On average, **77.2% of payouts** flow to other banks.  
        - This represents **millions in monthly leakage** that could be retained through targeted engagement.  
        """.format(high_risk_pct, avg_outflow))

        #Pie Chart
        # Data
        labels = ['Outside Bank', 'Within Bank']
        sizes = [77.2, 22.8]
        colors = ['red', 'green']

        # Create pie chart
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.pie(sizes, labels=labels,  autopct='%1.1f%%',colors=colors, startangle=90) 
        ax.set_title("Money Flow - Within vs Outside Bank")

        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["PctExternal"] * 100, bins=20, kde=False, color="skyblue", ax=ax)
        ax.axvline(50, color="red", linestyle="--", label="Risk Threshold (50%)")
        ax.set_title("Distribution of External Payout %", fontsize=12)
        ax.set_xlabel("External Payout (%)")
        ax.set_ylabel("Number of Corporates")
        ax.legend()
        st.pyplot(fig)

    # -------------------------
    # High-Risk Corporates
    # -------------------------
    with tab3:
        st.subheader("🚨 High-Risk Corporates")

        high_risk_df = df[df["OutflowRisk"] == "High"].sort_values(by="PctExternal", ascending=False).head(10)
        st.dataframe(high_risk_df[[
            "Corporate", "Industry", "Region", "CompanySize", "MonthlyVolume", "PctExternal"
        ]].style.format({
            "MonthlyVolume": "${:,.0f}",
            "PctExternal": "{:.0%}"
        }))

        st.markdown("""
        **Actionable Insights for Top High-Risk Payers:**  
        - Introduce **Liquidity Sweeps** to keep idle balances in-house.  
        - Offer **FX Hedging** to corporates with international transactions.  
        - Expand **Merchant Services** so counterparties settle internally.  
        - Propose **Virtual Card Numbers (VCN)** for supplier payments.  
        """)

    # -------------------------
    # What-If Simulation
    # -------------------------
    # --------------------------
    # Generate synthetic data
    # --------------------------
    np.random.seed(42)
    n_corporates = 50
    corporates = [f"Corp_{i+1}" for i in range(n_corporates)]

    data = pd.DataFrame({
        "Corporate": corporates,
        "Industry": np.random.choice(["Manufacturing", "IT", "Logistics", "Retail", "Healthcare"], n_corporates),
        "Size": np.random.choice(["SME", "Large"], n_corporates),
        "Region": np.random.choice(["East", "West", "North", "South"], n_corporates),
        "Tenure_Years": np.random.randint(1, 20, n_corporates),
        "Credit_Rating": np.random.choice(["AAA", "AA", "A", "BBB", "BB"], n_corporates),
        "Monthly_Payout": np.random.randint(1_000_000, 10_000_000, n_corporates),
        "Pct_External": np.random.uniform(20, 70, n_corporates),
        "Unique_Receivers": np.random.randint(10, 200, n_corporates),
        "Avg_Transaction_Size": np.random.randint(10_000, 500_000, n_corporates),
        "Products_Held": np.random.choice(["Operating Acct, FX", "Operating Acct, Payroll", 
                                        "Operating Acct, Merchant Services", 
                                        "Operating Acct, FX, Payroll"], n_corporates),
        "Product_Utilization": np.random.uniform(30, 100, n_corporates),
    })

    # Risk score: simulate probability
    data["Outflow_Risk_Score"] = (0.4*data["Pct_External"] +
                                0.3*(200 - data["Product_Utilization"]) +
                                0.3*(data["Unique_Receivers"]/2))
    data["Outflow_Risk_Prob"] = (data["Outflow_Risk_Score"] - data["Outflow_Risk_Score"].min()) / \
                                (data["Outflow_Risk_Score"].max() - data["Outflow_Risk_Score"].min())
    data["Risk_Level"] = pd.cut(data["Outflow_Risk_Prob"], bins=[0,0.33,0.66,1],
                                labels=["Low","Medium","High"])

    with tab4:
        st.subheader("Corporate Account Analysis")

        selected_corp = st.selectbox("Select a Corporate Account:", data["Corporate"].tolist())
        corp_data = data[data["Corporate"] == selected_corp].iloc[0]

            # Corporate Profile Table
        profile_data = {
            "Attribute": [
                "Industry",
                "Size",
                "Region",
                "Tenure (Years)",
                "Credit Rating",
                "Monthly Payout ($)",
                "External Payout %",
                "Unique Receivers",
                "Avg Transaction Size ($)",
                "Products Held",
                "Product Utilization %"
            ],
            "Value": [
                corp_data["Industry"],
                corp_data["Size"],
                corp_data["Region"],
                corp_data["Tenure_Years"],
                corp_data["Credit_Rating"],
                f"${corp_data['Monthly_Payout']:,}",
                f"{round(corp_data['Pct_External'],2)}%",
                corp_data["Unique_Receivers"],
                f"${corp_data['Avg_Transaction_Size']:,}",
                corp_data["Products_Held"],
                f"{round(corp_data['Product_Utilization'],1)}%"
            ]
        }

        profile_df = pd.DataFrame(profile_data)
        col1, col2 = st.columns(2)
        col1.metric("Predicted Outflow Risk Probability", f"{corp_data['Outflow_Risk_Prob']*100:.1f}%")
        col2.metric("Risk Level", corp_data["Risk_Level"])

        st.markdown("#### 📋 Corporate Profile")
        st.table(profile_df)



        # Visualization
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.pie([corp_data["Pct_External"], 100-corp_data["Pct_External"]],
                labels=["External", "Internal"], autopct='%1.1f%%', startangle=90)
            ax.set_title("Receiver Type Breakdown")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.bar(["Utilized", "Unutilized"], 
                [corp_data["Product_Utilization"], 100-corp_data["Product_Utilization"]])
            ax.set_title("Product Utilization")
            st.pyplot(fig)

        st.markdown("""
        **Recommended RM Actions:**  
        - Offer **Liquidity Sweep / VCN** to reduce external payouts.  
        - Cross-sell **FX hedging** if large share of payouts are overseas.  
        - Incentivize use of **Merchant Services** for counterparties.  
        - Strengthen engagement to **retain funds in-bank**.  
        """)




    # --------------------------
# USE CASE 2: Receiver Classification
# --------------------------
elif use_case == "Receiver Relationship Classification":
    st.title("🔗 Receiver Relationship Classification & Entity Resolution")

    st.markdown("""
    ### Business Objective  
    Corporates make payments to thousands of receivers.  
    Banks need to **resolve entities** and classify receivers to:  
    - Detect if a receiver is already an **in-bank customer under another account**.  
    - Classify receiver type (**supplier, payroll, tax authority, marketplace, etc.**).  
    - **Avoid redundant outreach** and tailor offers for retention.  
    """)

        # Receiver Classification Data
    n_receivers = 100
    n_corporates = 50
    corporates = [f"Corp_{i+1}" for i in range(n_corporates)]
    receivers = pd.DataFrame({
        "Receiver_ID": [f"R_{i+1}" for i in range(n_receivers)],
        "Corporate": np.random.choice(corporates, n_receivers),
        "Receiver_Name": [f"Receiver_{i+1}" for i in range(n_receivers)],
        "Is_InBank_Customer": np.random.choice([True, False], n_receivers, p=[0.4,0.6]),
        "Receiver_Type": np.random.choice(
            ["Supplier", "Payroll/Employee", "Tax Authority", "Marketplace", "Card Settlement", "Loan Payoff"], 
            n_receivers
        ),
        "Monthly_Transaction_Amt": np.random.randint(5_000, 500_000, n_receivers)
    })



    tab1, tab2, tab3 = st.tabs([
        "📊 Receiver Portfolio Overview", 
        "🔎 Receiver Drilldown",
        "💡 Potential Recovery"
    ])

    # Receiver Overview
    with tab1:
        st.subheader("Receiver Type Distribution")
        type_counts = receivers["Receiver_Type"].value_counts()

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title("Receiver Types")
            st.pyplot(fig)

        with col2:
            inbank_ratio = receivers["Is_InBank_Customer"].value_counts(normalize=True)*100
            fig, ax = plt.subplots()
            ax.bar(inbank_ratio.index.astype(str), inbank_ratio.values)
            ax.set_ylabel("% of Receivers")
            ax.set_title("In-bank vs External Receivers")
            st.pyplot(fig)

        st.markdown("""
        **Insights:**  
        - A large share of receivers are **external parties**, driving money outflow.  
        - Significant opportunity exists to **identify hidden in-bank links**.  
        """)

    # Receiver Drilldown
    with tab2:
        st.subheader("Receiver Drilldown")
        selected_receiver = st.selectbox("Select Receiver:", receivers["Receiver_ID"].tolist())
        r_data = receivers[receivers["Receiver_ID"] == selected_receiver].iloc[0]

        profile = {
            "Attribute": ["Corporate", "Receiver Name", "Is In-Bank Customer", "Receiver Type", "Monthly Txn Amt ($)"],
            "Value": [r_data["Corporate"], r_data["Receiver_Name"], r_data["Is_InBank_Customer"], 
                      r_data["Receiver_Type"], f"${r_data['Monthly_Transaction_Amt']:,}"]
        }
        st.table(pd.DataFrame(profile))

        st.markdown("#### 📈 Receiver Context")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.bar(["Txn Amount"], [r_data["Monthly_Transaction_Amt"]])
            ax.set_title("Receiver Monthly Flow")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.bar(["In-bank" if r_data["Is_InBank_Customer"] else "External"], [1], 
                   color="green" if r_data["Is_InBank_Customer"] else "red")
            ax.set_title("Receiver Classification")
            st.pyplot(fig)

        st.markdown("""
        **Recommended RM Actions:**  
        - If **in-bank customer**: Cross-link accounts, avoid redundant outreach.  
        - If **external supplier or payroll**: Offer **merchant services / payroll solutions**.  
        - If **tax authority / settlement**: Offer **cash management & liquidity services**.  
        """)

    # Business Impact Simulation
    with tab3:
        st.subheader("💡 Potential Recovery")
        st.markdown("""
        By resolving receiver identities, banks can discover that some **external receivers** are actually 
        **in-bank customers under different legal entities**.  
        
        Converting these flows back **reduces external outflow** and strengthens client stickiness.  
        """)

        # Assume 20% of external flows can be "recovered"
        external_flows = receivers[~receivers["Is_InBank_Customer"]]["Monthly_Transaction_Amt"].sum()
        potential_recovered = 0.2 * external_flows  

        st.metric("Total External Outflow (Monthly)", f"${external_flows:,.0f}")
        st.metric("Potential Recovery (20% of hidden in-bank links)", f"${potential_recovered:,.0f}")

        # Per-corporate recovery simulation
        recovery_df = receivers.groupby("Corporate")["Monthly_Transaction_Amt"].sum().reset_index()
        recovery_df["Potential_Recovery"] = recovery_df["Monthly_Transaction_Amt"] * 0.2  

        st.markdown("#### 📊 Corporate-wise Potential Recovery")
        st.dataframe(recovery_df.sort_values("Potential_Recovery", ascending=False).head(10))

        # Chart: Recovery Potential
        fig, ax = plt.subplots(figsize=(8,4))
        top10 = recovery_df.sort_values("Potential_Recovery", ascending=False).head(10)
        ax.barh(top10["Corporate"], top10["Potential_Recovery"])
        ax.set_xlabel("Potential Recovery ($)")
        ax.set_title("Top 10 Corporates by Recovery Potential")
        st.pyplot(fig)

        st.markdown("""
        **Business Takeaway:**  
        - Even identifying **20% of hidden in-bank relationships** can save millions monthly.  
        - Prioritizing **top corporates by recovery potential** ensures maximum impact.  
        - Supports RM strategy to **reduce leakage** and **tailor cross-sell offers**.  
        """)


# ============================
# Payment Network Intelligence
# (Streamlit code block — add as a tab / call as a function)
# Shows synthetic payment network, identifies "hub" external receivers,
# and renders a professional interactive network visualization (pyvis)
# plus supporting charts & actionable recommendations.
# ============================

elif use_case == "Payment Network Intelligence":

    def payment_network_intelligence_enhanced():
        st.header("🔎 Payment Network Intelligence — Identify Receiver Hubs")
        st.markdown("""
        **Objective:** Detect high-value *external* receivers (“hubs”) and visualize payments from corporates.
        Insights help prioritize engagement, merchant services, and fund retention.
        """)

        # ------------------------
        # Synthetic Data Generation
        # ------------------------
        np.random.seed(42)
        n_corp = 60
        n_recv = 120
        avg_links = 6
        hub_factor = 3

        corporates = [f"C_{i+1:03d}" for i in range(n_corp)]
        receivers = [f"R_{i+1:03d}" for i in range(n_recv)]
        attractiveness = (np.random.pareto(a=hub_factor, size=n_recv) + 1.0)
        receiver_prob = attractiveness / attractiveness.sum()

        receiver_types = ["Supplier","Payroll/Employee","Marketplace","Card Settlement","Tax Authority","Loan Payoff"]
        recv_meta = pd.DataFrame({
            "Receiver": receivers,
            "Receiver_Type": np.random.choice(receiver_types, size=n_recv),
            "Is_InBank": np.random.choice([True, False], size=n_recv, p=[0.25,0.75])
        })

        corp_monthly = np.random.lognormal(mean=12.5, sigma=0.7, size=n_corp)
        corp_monthly = corp_monthly / corp_monthly.mean() * 800_000

        edges = []
        for i, corp in enumerate(corporates):
            k = max(1, int(np.random.poisson(avg_links)))
            k = min(k, n_recv)
            chosen = np.random.choice(receivers, size=k, replace=False, p=receiver_prob)
            shares = np.random.dirichlet(alpha=np.ones(k))
            for r, s in zip(chosen, shares):
                amount = float(corp_monthly[i]*s*np.random.uniform(0.6,1.0))
                edges.append((corp,r,amount))
        edges_df = pd.DataFrame(edges, columns=["Corporate","Receiver","Amount"])
        edges_df = edges_df.groupby(["Corporate","Receiver"], as_index=False).agg({"Amount":"sum"})
        recv_stats = edges_df.groupby("Receiver").agg(
            NumPayers=("Corporate","nunique"),
            TotalReceived=("Amount","sum")
        ).reset_index().merge(recv_meta, left_on="Receiver", right_on="Receiver", how="left")

        payers_threshold = np.percentile(recv_stats["NumPayers"],90)
        amount_threshold = np.percentile(recv_stats["TotalReceived"],90)
        recv_stats["Is_Hub"] = (recv_stats["NumPayers"]>=payers_threshold)|(recv_stats["TotalReceived"]>=amount_threshold)

        # ------------------------
        # Corporate Filter Inside Tab
        # ------------------------
        st.subheader("Filter by Corporate Account")
        corp_filter = st.selectbox("Select Corporate (All = show full network):", ["All Corporates"] + corporates)
        if corp_filter != "All Corporates":
            edges_df_filtered = edges_df[edges_df["Corporate"]==corp_filter].copy()
            involved_receivers = edges_df_filtered["Receiver"].unique()
            recv_stats_filtered = recv_stats[recv_stats["Receiver"].isin(involved_receivers)].copy()
            corp_list = [corp_filter]
        else:
            edges_df_filtered = edges_df.copy()
            recv_stats_filtered = recv_stats.copy()
            corp_list = corporates

        # ------------------------
        # Summary Metrics
        # ------------------------
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        total_flow = edges_df_filtered["Amount"].sum()
        external_count = (~recv_stats_filtered["Is_InBank"]).sum()
        external_flow = recv_stats_filtered.loc[~recv_stats_filtered["Is_InBank"],"TotalReceived"].sum()
        col1.metric("Total Network Flow", f"${total_flow:,.0f}")
        col2.metric("External Receivers", f"{external_count:,}")
        col3.metric("Estimated External Flow", f"${external_flow:,.0f}")

        # ------------------------
        # Network Graph (Full Width)
        # ------------------------
        st.markdown("### Interactive Payment Network")
        st.markdown("""
        - Blue = Corporate Account
        - Green = In-bank Receiver  
        - Red = External Receiver  
        - Gold border = Hub  
        - Node size proportional to total received; edge thickness proportional to amount
        """)

        net = Network(height="800px", width="100%", bgcolor="#f8f9fa", font_color="#111111", directed=True)
        net.force_atlas_2based()

        # Add corporates
        for corp in corp_list:
            net.add_node(corp, label=corp, title=f"Corporate: {corp}", color="#1f78b4", shape="dot", size=12)
        # Add receivers
        min_amt = recv_stats_filtered["TotalReceived"].min()
        max_amt = recv_stats_filtered["TotalReceived"].max()
        for _, r in recv_stats_filtered.iterrows():
            size = 8 + 32*((r["TotalReceived"]-min_amt)/(max_amt-min_amt))
            color = "#2ca02c" if r["Is_InBank"] else "#d62728"
            net.add_node(r["Receiver"], label=r["Receiver"],
                        title=f"{r['Receiver']}<br>Type: {r['Receiver_Type']}<br>In-Bank: {r['Is_InBank']}<br>#Payers: {r['NumPayers']}<br>Total Received: ${r['TotalReceived']:,.0f}",
                        color=color, size=size, borderWidth=3 if r["Is_Hub"] else 0)
        # Add edges
        scale = max(1, edges_df_filtered["Amount"].max()/200_000)
        for row in edges_df_filtered.itertuples():
            net.add_edge(row.Corporate, row.Receiver, value=max(1,int(row.Amount/scale)), title=f"${row.Amount:,.0f}")

        net.show_buttons(filter_=['physics'])
        tmpfile = "payment_network.html"
        net.save_graph(tmpfile)
        with open(tmpfile,'r',encoding='utf-8') as f:
            html=f.read()
        components.html(html,height=850, scrolling=True)

        # ------------------------
        # Top Hubs Table & Charts
        # ------------------------
        st.subheader("Top Receiver Hubs & Recommendations")
        top_hubs = recv_stats_filtered.sort_values(["NumPayers","TotalReceived"], ascending=False).head(10).copy()
        top_hubs["TotalReceived_fmt"] = top_hubs["TotalReceived"].map(lambda x:f"${x:,.0f}")
        top_hubs["Action"] = top_hubs.apply(lambda r: (
            "Engage & onboard (high distinct payers)" if r["NumPayers"]>=payers_threshold else
            "Commercial partnership / merchant services"
        ), axis=1)
        st.table(top_hubs[["Receiver","Receiver_Type","Is_InBank","NumPayers","TotalReceived_fmt","Action"]].rename(columns={"Is_InBank":"In-Bank?"}))

        # Charts
        col1,col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7,3))
            ax.barh(top_hubs["Receiver"], top_hubs["NumPayers"], color="#ff7f0e")
            ax.set_xlabel("Distinct Corporate Payers")
            ax.set_title("Top Hubs by # of Payers")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7,3))
            ax.barh(top_hubs["Receiver"], top_hubs["TotalReceived"], color="#9467bd")
            ax.set_xlabel("Total Received ($)")
            ax.set_title("Top Hubs by Total Inflow")
            st.pyplot(fig)

        st.markdown("""
        **Insights / Recommendations:**  
        - Hubs with many payers → network engagement & merchant services.  
        - Hubs with high inflow → treasury solutions (FX, liquidity).  
        - External hubs → onboarding + KYC to retain flows.  
        - Corporate-level filter allows targeted analysis while maintaining full network visibility.
        """)
    payment_network_intelligence_enhanced()


    # -----------------------------
    # Implementation Requirements
    # -----------------------------



elif use_case == "Implementation Requirements":

    st.title("Imeplementation Requirements & Approach")

    # Dropdown for use case selection
    use_case = st.selectbox(
        "Select a Use Case:",
        [
            "Outflow Risk Forecast",
            "Payment Network Intelligence",
            "Receiver Classification & Entity Resolution"
        ]
    )

    # -----------------------------
    # Outflow Risk Forecast
    # -----------------------------
    if use_case == "Outflow Risk Forecast":
        
        def main():
    #st.set_page_config(page_title="Outflow Risk Forecast - ML Approach", layout="wide")

            st.title("💳 Outflow Risk Forecast (Payer-Level)")

            st.markdown("""
            ### Business Objective
            Predict the probability that **≥40–60% of a corporate’s balance/credits**  
            will flow to **external banks** in the next **T days (7/30 days)**.  

            **Use Case for Bank:**  
            - Identify **high-risk corporates** early.  
            - Enable **Relationship Managers** to intervene.  
            - Position targeted products: *Liquidity sweeps, Virtual Cards, FX hedging, Merchant services*.  
            """)

            # Tabs for structured view
            tab1, tab2, tab3, tab4 = st.tabs([
                "📌 Problem Statement", 
                "📊 Data Requirements", 
                "📂 Possible Data Sources", 
                "🤖 ML Approach"
            ])

            # -----------------------------
            # Problem Statement
            # -----------------------------
            with tab1:
                st.subheader("Problem Statement")
                st.markdown("""
                For each **corporate payer**, predict the probability that their  
                **external payouts (External Payouts ÷ Total Debits)** will exceed  
                **X% (e.g., 50%) over the next T days (7/30 days variants).**

                **Label Definition:**  
                - `1` → High Outflow Risk (≥ X%)  
                - `0` → Low/Moderate Outflow Risk (< X%)  

                **Business Value:**  
                - Retain funds within the bank  
                - Proactive engagement with corporates  
                - Revenue growth through tailored solutions  
                """)

                # -----------------------------
            # Data Requirements
            # -----------------------------
            with tab2:
                st.subheader("Data Requirements")

                feature_categories = {
                    "Corporate Profile": [
                        "Industry sector (Manufacturing, IT, Logistics, etc.)",
                        "Company size (SME, Large Corp, Multinational)",
                        "Region / Branch",
                        "Customer tenure with bank",
                        "Credit rating / internal risk score",
                        "Ownership type (Public, Private, Family-owned, PE-backed)",
                        "Parent-subsidiary relationship"
                    ],
                    "Transaction Behavior": [
                        "Historical monthly payout volume ($)",
                        "% of payouts to external vs internal accounts",
                        "No. of unique receivers (internal vs external)",
                        "Average transaction size",
                        "Max transaction size (outlier check)",
                        "Frequency of large-value payments",
                        "Day-of-week / Month-of-year seasonality",
                        "Volatility of monthly payout % (variance)"
                    ],
                    "Bank Product Usage": [
                        "Products held (Operating account, FX, Payroll, Merchant services, Treasury, VCN)",
                        "Product utilization % (active vs inactive)",
                        "Cross-sell index (number of distinct products used)",
                        "Trend in product adoption (increasing / declining usage)",
                        "Liquidity buffers maintained in bank"
                    ],
                    "Receiver Behavior": [
                        "Receiver type (supplier, payroll, tax authority, marketplace, etc.)",
                        "Receiver concentration (are payouts concentrated to few hubs?)",
                        "Top 5 receivers % share of total outflows",
                        "Receiver churn (new vs repeated receivers month-on-month)",
                        "Receiver’s domicile (domestic vs cross-border)"
                    ],
                    "Cash Flow & Liquidity Indicators": [
                        "Average account balance",
                        "Minimum balance maintained",
                        "Credit line utilization %",
                        "Overdraft / intraday liquidity usage",
                        "Incoming vs outgoing flow ratio"
                    ],
                    "External Market / Macroeconomic Factors": [
                        "Industry growth/decline trend",
                        "Macroeconomic stress indicators (inflation, interest rates)",
                        "Foreign exchange exposure (for cross-border payers)",
                        "Sector-specific payment risk indices"
                    ]
                }

                for category, feats in feature_categories.items():
                    st.markdown(f"**{category}:**")
                    df = pd.DataFrame({"Required Features": feats})
                    st.table(df)

            # -----------------------------
            # Data Sources
            # -----------------------------
            with tab3:
                st.subheader("Possible Data Sources")

                sources = {
                    "Internal Bank Systems": [
                        "Core Banking System (transaction debits/credits)",
                        "Customer Information File (CIF)",
                        "Internal credit risk rating system",
                        "Product utilization logs"
                    ],
                    "External Data Providers": [
                        "Credit Bureaus (Dun & Bradstreet, Experian, Equifax)",
                        "Market/industry datasets",
                        "Tax authority filings / public financial statements"
                    ],
                    "Derived Features": [
                        "Seasonality indicators (derived from transaction history)",
                        "Receiver network centrality (graph features from transaction network)",
                        "Concentration indices (Herfindahl-Hirschman Index for payouts)"
                    ]
                }

                for category, feats in sources.items():
                    st.markdown(f"**{category}:**")
                    df = pd.DataFrame({"Data Source / Input": feats})
                    st.table(df)

            # -----------------------------
            # ML Approach
            # -----------------------------
            with tab4:
                st.subheader("Machine Learning Approach")

                st.markdown("""
                ### Step 1: Problem Framing
                - Define label: `High Outflow Risk (≥X%) vs Low Risk`.  
                - Predict probability rather than binary classification (gives RM risk ranking).  

                ### Step 2: Data Preparation
                - Aggregate **payer-level transaction history** over rolling windows.  
                - Generate **features** from profile, behavior, product usage, and receivers.  
                - Handle imbalanced classes (SMOTE, class weights).  

                ### Step 3: Model Candidates
                - **Logistic Regression** (baseline, interpretable)  
                - **Gradient Boosted Trees (XGBoost, LightGBM)** for non-linear relationships  
                - **Neural Nets** if deep transaction features included  

                ### Step 4: Model Evaluation
                - Metrics: **AUC, Precision/Recall, F1, KS-statistic**  
                - Evaluate performance across **industries, sizes, and regions**  

                ### Step 5: Deployment & Usage
                - Integrate into RM dashboard  
                - Trigger **alerts** for high-risk corporates  
                - Recommend **next-best-action product offers**  
                """)
        main()


    # -----------------------------
    # Payment Network Intelligence
    # -----------------------------
    elif use_case == "Payment Network Intelligence":
        def main():
        #st.set_page_config(page_title="Payment Network Intelligence - Requirements", layout="wide")

            st.title("🌐 Payment Network Intelligence")

            st.markdown("""
            ### Business Objective
            Identify **key external receivers (“hubs”)** to whom multiple corporate payers  
            send funds, and detect **patterns of money flow concentration**.  

            **Use Case for Bank:**  
            - Detect high-value external payment hubs (e.g., payroll processors, tax authorities, large suppliers).  
            - Prioritize counterparties for **strategic partnerships**.  
            - Strengthen bank’s positioning in **receivables, merchant acquiring, and FX products**.  
            """)

            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📌 Problem Statement",
                "📊 Data Requirements",
                "📂 Data Platforms",
                "⚙️ Technology & Implementation"
            ])

            # -----------------------------
            # Problem Statement
            # -----------------------------
            with tab1:
                st.subheader("Problem Statement")
                st.markdown("""
                - Build a **sender–receiver payment network graph** where:  
                - Nodes = Corporate payers and receivers (internal & external).  
                - Edges = Payment flows (amount, frequency, product type).  

                - Use graph analytics / ML to answer:  
                1. Which receivers are **central hubs** (high in-degree, high value)?  
                2. Are payouts **concentrated to a few external nodes**?  
                3. Can we **predict network-level risk** (e.g., hub receiver default, fraud risk)?  

                **Business Value:**  
                - Identify high-value external entities → target them with **bank acquiring/receivables solutions**.  
                - Detect **concentration risk** in outflows.  
                - Enhance **cross-sell** (merchant services, VCN, FX) by leveraging network relationships.  
                """)

            # -----------------------------
            # Data Requirements
            # -----------------------------
            with tab2:
                st.subheader("Data Requirements")

                feature_categories = {
                    "Transaction Data": [
                        "Payer account ID",
                        "Receiver account ID (internal/external)",
                        "Receiver entity details (name, legal ID, domicile)",
                        "Transaction date & timestamp",
                        "Transaction amount",
                        "Currency",
                        "Transaction type (payroll, supplier, tax, settlement)",
                        "Channel (ACH, SWIFT, RTGS, card)"
                    ],
                    "Corporate Profile (Payer)": [
                        "Industry sector",
                        "Company size (SME, Large Corp, Multinational)",
                        "Region / Branch",
                        "Tenure with bank",
                        "Credit rating / risk score"
                    ],
                    "Receiver Profile": [
                        "Entity type (supplier, payroll processor, marketplace, tax authority, settlement)",
                        "Legal name / alias mapping (entity resolution)",
                        "Receiver’s relationship with bank (in-bank vs external)",
                        "Receiver concentration index (share of total inflows)"
                    ],
                    "Derived Network Features": [
                        "In-degree (number of distinct payers sending to receiver)",
                        "Out-degree (number of distinct receivers for a payer)",
                        "Edge weights (payment volume, frequency)",
                        "Centrality metrics (PageRank, betweenness, closeness)",
                        "Community detection (clusters of corporates and common receivers)"
                    ]
                }

                for category, feats in feature_categories.items():
                    st.markdown(f"**{category}:**")
                    df = pd.DataFrame({"Required Features": feats})
                    st.table(df)

            # -----------------------------
            # Data Platforms
            # -----------------------------
            with tab3:
                st.subheader("Data Platforms & Sources")

                sources = {
                    "Internal Bank Systems": [
                        "Core Banking System (payments ledger)",
                        "SWIFT/ACH/RTGS message logs",
                        "Merchant acquiring systems",
                        "Corporate product usage (Payroll, FX, Treasury)"
                    ],
                    "External / Third-Party Data": [
                        "Business registry databases (for receiver legal IDs)",
                        "Credit bureaus (receiver creditworthiness)",
                        "Industry payment directories / trade registries"
                    ],
                    "Data Lake / Warehousing": [
                        "Enterprise Data Lake (raw transaction storage)",
                        "Data Warehouse (curated payer–receiver datasets)",
                        "Streaming platforms (Kafka) for near real-time processing"
                    ],
                    "Specialized Stores": [
                        "Graph Database (Neo4j, TigerGraph, AWS Neptune, Azure Cosmos DB)",
                        "Vector DB (for entity resolution + embeddings)"
                    ]
                }

                for category, feats in sources.items():
                    st.markdown(f"**{category}:**")
                    df = pd.DataFrame({"Data Source / Platform": feats})
                    st.table(df)

            # -----------------------------
            # Technology & Implementation
            # -----------------------------
            with tab4:
                st.subheader("Technology & Implementation Approach")

                st.markdown("""
                ### Step 1: Data Engineering
                - Extract payment transactions from **core banking & SWIFT logs**.  
                - Standardize payer/receiver identifiers (handle duplicates, aliases).  
                - Load into **graph database** for efficient network queries.  

                ### Step 2: Network Graph Construction
                - Nodes: Corporate payers & receivers  
                - Edges: Payment flow with attributes (amount, frequency, product type)  
                - Derived metrics: **degree, centrality, community detection**  

                ### Step 3: Machine Learning / Analytics
                - **Graph analytics** to detect hubs (high in-degree / weighted centrality)  
                - **Entity resolution** to merge duplicate receiver identities  
                - **Risk classification** for external hubs (fraud risk, outflow concentration)  

                ### Step 4: Technology Stack
                - Data Ingestion: **Kafka, Spark Structured Streaming**  
                - Storage: **Data Lake (S3/HDFS)**, **Warehouse (Snowflake, BigQuery, Redshift)**  
                - Graph Processing: **Neo4j / TigerGraph / NetworkX for prototyping**  
                - Visualization: **Streamlit / Plotly / D3.js for interactive network graphs**  

                ### Step 5: Business Consumption
                - RM dashboards highlighting **hub receivers**  
                - Alerts for **new high-value external hubs**  
                - Insights for **strategic product positioning** (receivables, FX, merchant services)  
                """)
        main()

    # -----------------------------
    # Receiver Classification & Entity Resolution
    # -----------------------------
    elif use_case == "Receiver Classification & Entity Resolution":

        st.title("Receiver Relationship Classification & Entity Resolution")

        # Tabs
        tabs = st.tabs(["Problem Statement", "Data Requirements", "Platforms & Technology", "Solution Approach"])

        # -------------------------------
        # Problem Statement
        # -------------------------------
        with tabs[0]:
            st.header("Problem Statement")
            st.markdown("""
            Banks process millions of outbound payments from corporate clients to receivers.  
            Key challenges include:
            - **Entity Resolution:** Identifying whether a receiver is already an in-bank customer under a different legal name or account.  
            - **Receiver Classification:** Categorizing receivers into business-relevant categories (e.g., supplier, payroll, marketplace, tax authority).  

            **Business Objective**  
            - Avoid redundant outreach by recognizing existing customer relationships.  
            - Tailor product offers (e.g., liquidity, FX, payables solutions) to receiver categories.  
            - Strengthen customer retention by providing insights into payment flows.  
            """)

        # -------------------------------
        # Data Requirements
        # -------------------------------
        with tabs[1]:
            st.header("Data Requirements")
            st.subheader("Receiver Identification & Enrichment")
            st.markdown("""
            - **Payment Transaction Data**
                - Sender Account ID, Receiver Name, Receiver Account Number/IBAN
                - Transaction Date, Amount, Currency
                - Payment Channel (SWIFT, ACH, RTP, Card, etc.)
            - **Customer Master Data**
                - In-bank corporate customer names, aliases, legal entity IDs, account numbers
            - **External Data Sources**
                - Business registry data (DUNS, LEI, Company House, UBO registries)
                - Tax authority databases
            - **Textual Data**
                - Payment remittance notes
                - Invoice/payment references
                - Memo/description fields
            """)

            st.subheader("Feature Categories")
            st.markdown("""
            - **Entity Resolution Features**
                - Name similarity (string distance, phonetic similarity, fuzzy matching)
                - Account number / IBAN cross-mapping
                - Legal identifiers (LEI, VAT ID, Company Registration No.)
            - **Receiver Classification Features**
                - Remittance text keywords (e.g., *Payroll, Invoice, Tax, Amazon*)
                - Named Entity Recognition (organizations, people, government authorities)
                - Embeddings from FastText/Word2Vec for semantic similarity
                - Frequency & transaction behavior patterns
            """)

        # -------------------------------
        # Platforms & Technology
        # -------------------------------
        with tabs[2]:
            st.header("Platforms & Technology Requirements")
            st.markdown("""
            **Data Platforms**
            - Enterprise Data Lake (stores transaction + customer master data)
            - External Data Providers: OpenCorporates, GLEIF (LEI), DUN & Bradstreet
            - NLP Enrichment APIs for Named Entity Recognition  

            **Technologies**
            - **Entity Resolution**
                - FuzzyWuzzy, RapidFuzz, or Python `difflib` for string matching
                - Phonetic algorithms (Soundex, Metaphone)
                - Graph-based linkage (Neo4j or NetworkX for entity linking)
            - **Receiver Classification**
                - Rule-based keyword matching
                - NLP Models: spaCy, HuggingFace Transformers (NER)
                - Embedding Models: FastText, Word2Vec, BERT embeddings
                - ML Classifiers: Logistic Regression, XGBoost, or Neural Networks
            - **Deployment**
                - Streamlit/Flask for prototype
                - Scalable microservices with REST APIs
                - Model serving on MLflow/Sagemaker
            """)

        # -------------------------------
        # Solution Approach
        # -------------------------------
        with tabs[3]:
            st.header("Solution Approach")

            st.subheader("Step 1: Entity Resolution")
            st.markdown("""
            - Match receiver name/account against in-bank customer database.  
            - Techniques:
                - Fuzzy string matching
                - Alias dictionary & phonetic encoding
                - Graph-based entity linkage
            - Output: **Match / No-Match + Confidence Score**
            """)

            st.subheader("Step 2: Receiver Classification")
            st.markdown("""
            **Three-step layered classification:**

            1. **Keyword-Based Rules**
            - Match remittance text against domain-specific keywords.  
            - Example: *Payroll → Employee/Salary, Tax → Government Authority, Invoice → Supplier*.

            2. **Named Entity Recognition (NER)**
            - Use NLP to detect organization, government authority, marketplace names.  
            - Example: *Amazon → Marketplace, IRS → Tax Authority*.

            3. **Embedding + ML Classification**
            - Train embedding model (FastText/Word2Vec) on remittance text.  
            - Classify receiver into categories: Supplier, Payroll, Tax, Marketplace, Loan Payoff, etc.  
            """)

            st.subheader("Step 3: Business Usage")
            st.markdown("""
            - **Avoid Redundant Outreach**: If receiver is already an in-bank client → no duplicate marketing.  
            - **Tailored Offers**:  
                - Payroll Receivers → Payroll cards, salary processing solutions  
                - Suppliers → Trade finance, supply chain lending  
                - Marketplaces → Merchant acquiring, FX services  
            - **Portfolio Analytics**: Identify dominant receiver categories for a corporate → product strategy.  
            """)





