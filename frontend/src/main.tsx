import React, { Suspense, lazy } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import App from "./App";
import Overview from "./pages/Overview";
import RequestDetail from "./pages/RequestDetail";
import Requests from "./pages/Requests";
import "./styles.css";

// Benchmarks pulls in the charting library; keep it out of the initial bundle.
const Benchmarks = lazy(() => import("./pages/Benchmarks"));

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<Overview />} />
          <Route path="requests" element={<Requests />} />
          <Route path="requests/:id" element={<RequestDetail />} />
          <Route
            path="benchmarks"
            element={
              <Suspense fallback={<p className="empty">Loading…</p>}>
                <Benchmarks />
              </Suspense>
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
