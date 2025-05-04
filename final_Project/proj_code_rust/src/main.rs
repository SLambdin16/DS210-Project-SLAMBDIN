use proj_code::{Graph, find_clusters, load_correlation_matrix};
use csv::{ReaderBuilder, Writer};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<(), Box<dyn Error>> {
    // Load correlation matrix and get tickers
    let (tickers, matrix) = load_correlation_matrix("correlation_matrix.csv")?;

    // Construct the graph
    let threshold = 0.7;
    let graph = Graph::from_correlation_matrix(&matrix, threshold);
    println!("Graph constructed with {} nodes.", tickers.len());

    // Find clusters
    let clusters = find_clusters(&graph);
    println!("Identified {} clusters.\n", clusters.len());

    // --- Merge Logic ---
    let ticker_sector_path = "ticker_sector_map.csv";
    let mut ticker_to_sector: HashMap<String, String> = HashMap::new();

    // Read ticker-sector mapping
    let ticker_file = File::open(ticker_sector_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(BufReader::new(ticker_file));

    for result in rdr.records() {
        let record = result?;
        let ticker = record.get(1).unwrap().trim().to_string(); // 'name' column
        let sector = record.get(2).unwrap().trim().to_string(); // 'Sector' column
        ticker_to_sector.insert(ticker, sector);
    }

    // Prepare output CSV
    let output_path = "clustered_sectors2.csv";
    let mut wtr = Writer::from_path(output_path)?;
    wtr.write_record(&["cluster", "ticker", "sector"])?;

    // Write cluster-ticker-sector data
    for (cluster_id, cluster) in clusters.iter().enumerate() {
        for &stock_index in cluster {
            if let Some(ticker) = tickers.get(stock_index) {
                if let Some(sector) = ticker_to_sector.get(ticker) {
                    // Corrected line:
                    wtr.write_record(&[cluster_id.to_string(), ticker.to_string(), sector.to_string()])?;
                } else {
                    eprintln!("Warning: Ticker {} not found in sector map.", ticker);
                }
            } else {
                eprintln!("Warning: Index {} out of bounds for tickers.", stock_index);
            }
        }
    }

    wtr.flush()?;
    println!("Successfully merged cluster and sector data to {}", output_path);

    Ok(())
}