import { useState } from 'react';
import { apiJson, displayText, reportSeedHtml } from '../utils/helpers';

export function useLoadLatest() {
  const [latest, setLatest] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [feedback, setFeedback] = useState({ records: [], summary: { count: 0 } });
  const [observability, setObservability] = useState(null);
  const [loading, setLoading] = useState(false);

  const loadLatest = async (quiet = false) => {
    setLoading(true);
    try {
      const data = await apiJson('/api/coordinator/latest');
      setLatest(data.output || null);
      setMetadata(data.metadata || null);
      setFeedback(data.feedback || { records: [], summary: { count: 0 } });
      setObservability(data.observability || null);
      return data;
    } catch (error) {
      if (!quiet) {
        const { message: msg } = await import('antd');
        msg.warning(error.message || 'No completed analysis is available');
      }
    } finally {
      setLoading(false);
    }
  };

  return { latest, setLatest, metadata, setMetadata, feedback, setFeedback, observability, setObservability, loading, loadLatest };
}

export function useLoadStatus() {
  const [system, setSystem] = useState({ started: false, starting: false });
  const [reportStatus, setReportStatus] = useState(null);
  const [config, setConfig] = useState({});

  const loadStatus = async () => {
    try {
      const [systemData, reportData, configData] = await Promise.allSettled([
        apiJson('/api/system/status'),
        apiJson('/api/report/status'),
        apiJson('/api/config')
      ]);
      if (systemData.status === 'fulfilled') setSystem(systemData.value);
      if (reportData.status === 'fulfilled') setReportStatus(reportData.value);
      if (configData.status === 'fulfilled') setConfig(configData.value.config || {});
    } catch {
      return undefined;
    }
  };

  return { system, setSystem, reportStatus, setReportStatus, config, setConfig, loadStatus };
}

export function useObservabilityTrace() {
  const [observabilityTrace, setObservabilityTrace] = useState(null);
  const [loading, setLoading] = useState(false);

  const loadObservabilityTrace = async (quiet = true) => {
    setLoading(true);
    try {
      const data = await apiJson('/api/observability/langsmith');
      setObservabilityTrace(data);
      if (!quiet) {
        const { message: msg } = await import('antd');
        msg.success(data.source === 'langsmith' ? 'Traces loaded' : 'Local trace loaded');
      }
    } catch (error) {
      if (!quiet) {
        const { message: msg } = await import('antd');
        msg.warning(error.message || 'Trace data is pending');
      }
    } finally {
      setLoading(false);
    }
  };

  return { observabilityTrace, setObservabilityTrace, loading, loadObservabilityTrace };
}
