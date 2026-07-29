// A broad universe of large/mid-cap Japanese stocks (Nikkei 225 constituents),
// spread across sectors so the landscape has real structure. Tickers are Tokyo
// Stock Exchange codes with the `.T` suffix used by Yahoo Finance. Failed/illiquid
// tickers are skipped gracefully at fetch time, so an occasional stale code is fine.

const RAW = [
  // autos & parts
  "7203.T", "7267.T", "7201.T", "7269.T", "7270.T", "7211.T", "7202.T", "7205.T", "6902.T", "7259.T", "6201.T", "5108.T",
  // electronics / precision / machinery
  "6758.T", "6501.T", "6752.T", "6702.T", "6971.T", "6981.T", "6954.T", "6503.T", "6367.T", "7751.T", "6479.T", "6506.T",
  "6645.T", "6841.T", "6952.T", "6861.T", "7741.T", "7733.T", "4543.T", "7832.T", "7912.T", "6273.T", "6146.T",
  // semiconductors
  "8035.T", "6857.T", "6723.T", "6963.T", "4062.T", "6920.T", "3436.T",
  // telecom / it services / internet
  "9432.T", "9433.T", "9434.T", "9984.T", "4689.T", "9613.T", "4307.T", "9602.T", "2432.T", "3659.T", "4755.T",
  // banks / finance / insurance
  "8306.T", "8316.T", "8411.T", "8591.T", "8604.T", "8309.T", "8308.T", "8725.T", "8766.T", "8630.T", "8750.T", "7182.T",
  // trading houses
  "8058.T", "8031.T", "8001.T", "8053.T", "8002.T", "2768.T",
  // pharma / chemicals health
  "4502.T", "4503.T", "4568.T", "4523.T", "4519.T", "4578.T", "4507.T", "4151.T", "4506.T", "4901.T", "4911.T", "4452.T",
  // retail / consumer
  "9983.T", "3382.T", "8267.T", "2914.T", "3092.T", "7532.T", "8113.T", "2801.T", "2502.T", "2503.T", "2802.T", "2269.T",
  // machinery / heavy / capital goods
  "6301.T", "7011.T", "7012.T", "7013.T", "6326.T", "6305.T", "6113.T", "5631.T", "7004.T", "6104.T",
  // materials / steel / chemical / glass
  "4063.T", "4188.T", "5401.T", "3407.T", "5411.T", "5406.T", "4005.T", "4183.T", "5201.T", "5333.T", "3405.T",
  // transport / logistics / airlines
  "9020.T", "9022.T", "9101.T", "9104.T", "9107.T", "9201.T", "9202.T", "9147.T",
  // utilities / energy
  "9501.T", "9503.T", "9531.T", "5020.T", "1605.T", "5019.T",
  // real estate / construction
  "8801.T", "8802.T", "8830.T", "1801.T", "1802.T", "1803.T", "1928.T", "1925.T",
  // services / entertainment / other
  "6098.T", "4661.T", "7974.T", "9735.T", "2413.T", "4324.T", "9766.T", "4704.T", "6178.T",
];

export const UNIVERSE = [...new Set(RAW)];
