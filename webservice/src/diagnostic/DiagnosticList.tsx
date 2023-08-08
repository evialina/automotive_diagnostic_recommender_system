import { Datagrid, List, TextField } from 'react-admin';

const defaultItems = [
    { 'source': 'id', label: 'ID' },
    { 'source': 'anonymised_vin' , label: 'Anonymised VIN' },
    { 'source': 'module', label: 'Module' },
    { 'source': 'faulttype', label: 'Fault Type' },
    { 'source': 'dtcfull', label: 'DTC Full' },
    { 'source': 'dtcdescription', label: 'DTC Description' },
    { 'source': 'odomiles', label: 'Odometer Reading' },
    { 'source': 'sessiontimestamp', label: 'Session Timestamp' },
    { 'source': 'builddate', label: 'Build Date'  },
    { 'source': 'warrantydate', label: 'Warranty Start Date' },
]

export const DiagnosticList = ({ items = defaultItems, ...params }) => (
    <List {...params}>
        <Datagrid>
            {items.map(item => <TextField key={item.source} source={item.source} label={item?.label || item.source} />)}
        </Datagrid>
    </List>
)
