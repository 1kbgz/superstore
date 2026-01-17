use std::collections::HashMap;

lazy_static::lazy_static! {
    pub static ref US_SECTORS_MAP: HashMap<&'static str, Vec<&'static str>> = {
        let mut m = HashMap::new();
        m.insert("Consumer Discretionary", vec![
            "Auto Components",
            "Automobiles",
            "Distributors",
            "Diversified Consumer Services",
            "Hotels, Restaurants & Leisure",
            "Household Durables",
            "Internet & Catalog Retail",
            "Leisure Products",
            "Media",
            "Multiline Retail",
            "Specialty Retail",
            "Textiles, Apparel & Luxury Goods",
        ]);
        m.insert("Consumer Staples", vec![
            "Beverages",
            "Food & Staples Retailing",
            "Food Products",
            "Household Products",
            "Personal Products",
            "Tobacco",
        ]);
        m.insert("Energy", vec![
            "Energy Equipment & Services",
            "Oil, Gas & Consumable Fuels",
        ]);
        m.insert("Financials", vec![
            "Banks",
            "Capital Markets",
            "Consumer Finance",
            "Diversified Financial Services",
            "Insurance",
            "Mortgage REITs",
            "Thrifts & Mortgage Finance",
        ]);
        m.insert("Health Care", vec![
            "Biotechnology",
            "Health Care Equipment & Supplies",
            "Health Care Providers & Services",
            "Health Care Technology",
            "Life Sciences Tools & Services",
            "Pharmaceuticals",
        ]);
        m.insert("Industrials", vec![
            "Aerospace & Defense",
            "Air Freight & Logistics",
            "Airlines",
            "Building Products",
            "Commercial Services & Supplies",
            "Construction & Engineering",
            "Electrical Equipment",
            "Industrial Conglomerates",
            "Machinery",
            "Marine",
            "Professional Services",
            "Road & Rail",
            "Trading Companies & Distributors",
            "Transportation Infrastructure",
        ]);
        m.insert("Information Technology", vec![
            "Communications Equipment",
            "Electronic Equipment, Instruments & Components",
            "IT Services",
            "Internet Software & Services",
            "Semiconductors & Semiconductor Equipment",
            "Software",
            "Technology Hardware, Storage & Peripherals",
        ]);
        m.insert("Materials", vec![
            "Chemicals",
            "Construction Materials",
            "Containers & Packaging",
            "Metals & Mining",
            "Paper & Forest Products",
        ]);
        m.insert("Real Estate", vec![
            "Equity Real Estate Investment Trusts",
            "Real Estate Management & Development",
        ]);
        m.insert("Telecommunication Services", vec![
            "Diversified Telecommunication Services",
            "Wireless Telecommunication Services",
        ]);
        m.insert("Utilities", vec![
            "Electric Utilities",
            "Gas Utilities",
            "Independent Power and Renewable Electricity Producers",
            "Multi-Utilities",
            "Water Utilities ",
        ]);
        m
    };

    pub static ref US_SECTORS: Vec<&'static str> = US_SECTORS_MAP.keys().copied().collect();
}
