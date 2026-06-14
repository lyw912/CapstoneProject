import React from 'react';
import { Alert } from 'antd';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <Alert
          type="error"
          showIcon
          message="View crashed"
          description={this.state.error.message || 'An unexpected error occurred in this view.'}
        />
      );
    }
    return this.props.children;
  }
}
