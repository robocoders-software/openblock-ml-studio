import React from 'react';
import ReactDOM from 'react-dom';
import AppDialog from '../components/app-dialog/app-dialog.jsx';

let _container = null;

const getContainer = () => {
    if (_container && document.body.contains(_container)) return _container;
    _container = document.createElement('div');
    _container.id = 'robocoders-dialog-root';
    document.body.appendChild(_container);
    return _container;
};

const showAppDialog = ({
    type      = 'info',
    title     = '',
    message   = '',
    detail    = null,
    buttons   = ['OK'],
    defaultId = 0
} = {}) => new Promise(resolve => {
    const container = getContainer();

    const handleClick = idx => {
        ReactDOM.unmountComponentAtNode(container);
        resolve(idx);
    };

    ReactDOM.render(
        <AppDialog
            type={type}
            title={title}
            message={message}
            detail={detail}
            buttons={buttons}
            defaultId={defaultId}
            onButtonClick={handleClick}
        />,
        container
    );
});

export default showAppDialog;
