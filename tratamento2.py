
import time
from pathlib import Path

import numpy as np
import pandas as pd


ARQUIVO_ENTRADA = Path("dados_completos.xlsx")
ARQUIVO_SAIDA = Path("dados_tratados.xlsx")

LIMITE_CLASSE_A = 0.80
LIMITE_CLASSE_B = 0.95

COLS_PRODUTOS = [
    "id_produto", "nome_produto", "categoria", "preco_custo", "preco_venda",
    "lead_time_dias", "estoque_atual", "estoque_reserva", "media_diaria_ajustada", "ponto_pedido_ajustado"
]
COLS_VENDAS = ["id_venda", "id_produto", "quantidade_vendas", "data"]

MESES_PT = np.array([
    "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro",
])
DIAS_PT = np.array([
    "Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo",
])


def erro_amigavel(mensagem):
    raise SystemExit(f"[ERRO] {mensagem}")


def carregar(caminho):
    if not caminho.exists():
        erro_amigavel(f"Arquivo não encontrado: {caminho}")
    try:
        produtos = pd.read_excel(caminho, sheet_name="produtos")
        vendas = pd.read_excel(caminho, sheet_name="vendas")
    except ValueError:
        erro_amigavel("A planilha precisa ter as abas 'produtos' e 'vendas'.")

    for nome, df, cols in (("produtos", produtos, COLS_PRODUTOS),
                           ("vendas", vendas, COLS_VENDAS)):
        faltando = [c for c in cols if c not in df.columns]
        if faltando:
            erro_amigavel(f"Aba '{nome}' sem as colunas: {', '.join(faltando)}")

    return produtos[COLS_PRODUTOS].copy(), vendas[COLS_VENDAS].copy()


def tratar_produtos(df):
    df["id_produto"] = df["id_produto"].astype(str).str.strip().str.upper()
    df["nome_produto"] = (df["nome_produto"].astype(str).str.strip()
                          .str.replace(r"\s+", " ", regex=True))
    df["categoria"] = df["categoria"].astype(str).str.strip().str.lower().str.capitalize()

    cols_num = ["preco_custo", "preco_venda", "lead_time_dias",
                "estoque_atual", "estoque_reserva"]
    for c in cols_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    motivo = np.select(
        [
            df[cols_num].isna().any(axis=1) | df["id_produto"].isin(["", "NAN"]),
            df["nome_produto"].isin(["", "nan"]),
            (df[cols_num] < 0).any(axis=1),
            df["preco_venda"] < df["preco_custo"],
            df.duplicated("id_produto", keep="first"),
        ],
        [
            "campo obrigatório vazio ou inválido",
            "nome vazio",
            "valor negativo",
            "preço de venda menor que o custo",
            "id_produto duplicado",
        ],
        default="",
    )
    rejeitados = df[motivo != ""].assign(motivo_rejeicao=motivo[motivo != ""])
    validos = df[motivo == ""].copy()

    validos[["lead_time_dias", "estoque_atual", "estoque_reserva"]] = (
        validos[["lead_time_dias", "estoque_atual", "estoque_reserva"]].astype(int)
    )

    validos["nome_duplicado"] = validos.duplicated("nome_produto", keep=False)
    return validos, rejeitados


def tratar_vendas(df, ids_validos):
    df["id_venda"] = df["id_venda"].astype(str).str.strip().str.upper()
    df["id_produto"] = df["id_produto"].astype(str).str.strip().str.upper()
    df["quantidade_vendas"] = pd.to_numeric(df["quantidade_vendas"], errors="coerce")
    df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.normalize()

    motivo = np.select(
        [
            df["quantidade_vendas"].isna() | df["data"].isna(),
            df["quantidade_vendas"] <= 0,
            ~df["id_produto"].isin(ids_validos),
            df.duplicated("id_venda", keep="first"),
        ],
        [
            "quantidade ou data inválida",
            "quantidade menor ou igual a zero",
            "produto não cadastrado",
            "id_venda duplicado",
        ],
        default="",
    )
    rejeitados = df[motivo != ""].assign(motivo_rejeicao=motivo[motivo != ""])
    validas = df[motivo == ""].copy()
    validas["quantidade_vendas"] = validas["quantidade_vendas"].astype(int)
    return validas, rejeitados


def calcular_indicadores(produtos, vendas):
    qtd = (vendas.groupby("id_produto", as_index=False)["quantidade_vendas"].sum()
           .rename(columns={"quantidade_vendas": "qtd_vendida"}))

    df = produtos.merge(qtd, on="id_produto", how="left")
    df["qtd_vendida"] = df["qtd_vendida"].fillna(0).astype(int)

    df["faturamento_total"] = df["qtd_vendida"] * df["preco_venda"]
    df["lucro_bruto"] = df["qtd_vendida"] * (df["preco_venda"] - df["preco_custo"])

    dias = (vendas["data"].max() - vendas["data"].min()).days + 1
    df["vendas_media_diaria"] = (df["qtd_vendida"] / dias).round(4)

    df["ponto_pedido"] = (df["vendas_media_diaria"] * df["lead_time_dias"]
                          + df["estoque_reserva"]).round(2)

    df["status_estoque"] = np.select(
        [df["estoque_atual"] <= 0, df["estoque_atual"] <= df["ponto_pedido"]],
        ["ruptura", "comprar"],
        default="ok",
    )

    df = df.sort_values(["faturamento_total", "id_produto"],
                        ascending=[False, True]).reset_index(drop=True)
    total = df["faturamento_total"].sum()
    if total == 0:
        erro_amigavel("Faturamento total igual a zero: não é possível calcular a Curva ABC.")

    df["pct_faturamento"] = df["faturamento_total"] / total
    df["pct_acumulado"] = df["pct_faturamento"].cumsum().round(6)
    df["curva_abc"] = np.select(
        [df["pct_acumulado"] <= LIMITE_CLASSE_A, df["pct_acumulado"] <= LIMITE_CLASSE_B],
        ["A", "B"],
        default="C",
    )
    df["ranking_abc"] = np.arange(1, len(df) + 1)
    return df, dias


def montar_calendario(vendas):
    datas = pd.date_range(vendas["data"].min(), vendas["data"].max(), freq="D")
    return pd.DataFrame({
        "data": datas,
        "ano": datas.year,
        "mes_num": datas.month,
        "mes": MESES_PT[datas.month - 1],
        "ano_mes": datas.strftime("%Y-%m"),
        "trimestre": "T" + datas.quarter.astype(str),
        "dia_semana": DIAS_PT[datas.dayofweek],
    })


def montar_kpis(prod, vendas, dias):
    contagem_abc = prod["curva_abc"].value_counts()
    fat_abc = prod.groupby("curva_abc")["faturamento_total"].sum()
    linhas = [
        ("Produtos cadastrados", len(prod)),
        ("Produtos em ruptura", int((prod["status_estoque"] == "ruptura").sum())),
        ("Produtos para comprar (<= ponto de pedido)", int((prod["status_estoque"] == "comprar").sum())),
        ("Movimentações de saída no período", len(vendas)),
        ("Unidades vendidas", int(vendas["quantidade_vendas"].sum())),
        ("Faturamento total (R$)", round(prod["faturamento_total"].sum(), 2)),
        ("Lucro bruto total (R$)", round(prod["lucro_bruto"].sum(), 2)),
        ("Produtos classe A", int(contagem_abc.get("A", 0))),
        ("Produtos classe B", int(contagem_abc.get("B", 0))),
        ("Produtos classe C", int(contagem_abc.get("C", 0))),
        ("% do faturamento na classe A", round(fat_abc.get("A", 0) / prod["faturamento_total"].sum(), 4)),
        ("Período - início", vendas["data"].min().date().isoformat()),
        ("Período - fim", vendas["data"].max().date().isoformat()),
        ("Dias no período", dias),
    ]
    return pd.DataFrame(linhas, columns=["indicador", "valor"])


def main():
    inicio = time.perf_counter()

    produtos_raw, vendas_raw = carregar(ARQUIVO_ENTRADA)
    produtos, prod_rej = tratar_produtos(produtos_raw)
    vendas, vend_rej = tratar_vendas(vendas_raw, set(produtos["id_produto"]))
    prod_final, dias = calcular_indicadores(produtos, vendas)

    alertas = (prod_final[prod_final["status_estoque"] != "ok"]
               [["id_produto", "nome_produto", "categoria", "estoque_atual",
                 "ponto_pedido", "status_estoque", "curva_abc"]]
               .sort_values(["status_estoque", "curva_abc"], ascending=[False, True]))

    calendario = montar_calendario(vendas)
    kpis = montar_kpis(prod_final, vendas, dias)

    try:
        with pd.ExcelWriter(ARQUIVO_SAIDA, engine="openpyxl",
                            date_format="YYYY-MM-DD", datetime_format="YYYY-MM-DD") as xl:
            prod_final.to_excel(xl, sheet_name="produtos", index=False)
            vendas.to_excel(xl, sheet_name="vendas", index=False)
            calendario.to_excel(xl, sheet_name="calendario", index=False)
            alertas.to_excel(xl, sheet_name="alertas_estoque", index=False)
            kpis.to_excel(xl, sheet_name="kpis", index=False)
            if len(prod_rej) or len(vend_rej):
                prod_rej.to_excel(xl, sheet_name="rejeitados_produtos", index=False)
                vend_rej.to_excel(xl, sheet_name="rejeitados_vendas", index=False)
    except PermissionError:
        erro_amigavel(f"Feche o arquivo '{ARQUIVO_SAIDA}' no Excel e execute novamente.")

    print("=" * 60)
    print("TRATAMENTO CONCLUÍDO")
    print("=" * 60)
    print(f"Produtos: {len(produtos_raw)} lidos | {len(produtos)} válidos | {len(prod_rej)} rejeitados")
    print(f"Vendas  : {len(vendas_raw)} lidas | {len(vendas)} válidas | {len(vend_rej)} rejeitadas")
    print(f"Produtos com nome repetido (IDs diferentes): {int(produtos['nome_duplicado'].sum())}")
    print(f"Período: {vendas['data'].min().date()} a {vendas['data'].max().date()} ({dias} dias)")
    print("\nCurva ABC:")
    resumo = prod_final.groupby("curva_abc").agg(
        produtos=("id_produto", "count"),
        faturamento=("faturamento_total", "sum"),
    )
    resumo["% faturamento"] = (resumo["faturamento"] / resumo["faturamento"].sum() * 100).round(1)
    print(resumo.to_string())
    print("\nStatus de estoque:")
    print(prod_final["status_estoque"].value_counts().to_string())
    print(f"\nArquivo gerado: {ARQUIVO_SAIDA}")
    print(f"Tempo de execução: {time.perf_counter() - inicio:.2f}s")


if __name__ == "__main__":
    main()
