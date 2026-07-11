import { useState } from 'react';
import {
  Button,
  Drawer,
  Input,
  Select,
  Switch,
  message
} from 'antd';
import { PlayCircleOutlined } from '@ant-design/icons';
import { CONFIG_GROUPS } from '../utils/constants';
import { apiJson } from '../utils/helpers';

export default function ConfigDrawer({ open, onClose, config, setConfig, onSaved }) {
  const [saving, setSaving] = useState(false);
  const update = (key, value) => setConfig((current) => ({ ...current, [key]: value }));
  const save = async (startAfter = false) => {
    setSaving(true);
    try {
      const data = await apiJson('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      setConfig(data.config || config);
      if (startAfter) {
        await apiJson('/api/system/start', { method: 'POST' });
      }
      message.success(startAfter ? 'Configuration saved and system startup requested' : 'Configuration saved');
      onSaved?.();
      onClose();
    } catch (error) {
      message.error(error.message || 'Configuration update failed');
    } finally {
      setSaving(false);
    }
  };
  return (
    <Drawer open={open} onClose={onClose} width={620} title="Workspace Configuration" extra={<Button type="primary" loading={saving} onClick={() => save(false)}>Save</Button>}>
      <div className="drawer-stack">
        {CONFIG_GROUPS.map((group) => (
          <section className="config-card" key={group.title}>
            <h3>{group.title}</h3>
            <p>{group.description}</p>
            <div className="config-fields">
              {group.fields.map(([key, label, type]) => (
                <label key={key}>
                  <span>{label}</span>
                  {type === 'select' ? (
                    <Select value={config[key] || 'TavilyAPI'} onChange={(value) => update(key, value)} options={[{ value: 'TavilyAPI' }, { value: 'BochaAPI' }, { value: 'AnspireAPI' }]} />
                  ) : type === 'boolean' ? (
                    <Switch checked={String(config[key]).toLowerCase() === 'true'} onChange={(checked) => update(key, checked ? 'True' : 'False')} />
                  ) : type === 'password' ? (
                    <Input.Password value={config[key] || ''} onChange={(event) => update(key, event.target.value)} />
                  ) : (
                    <Input value={config[key] || ''} onChange={(event) => update(key, event.target.value)} />
                  )}
                </label>
              ))}
            </div>
          </section>
        ))}
        <Button block size="large" type="primary" icon={<PlayCircleOutlined />} loading={saving} onClick={() => save(true)}>Save and Start Runtime</Button>
      </div>
    </Drawer>
  );
}
